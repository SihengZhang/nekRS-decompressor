#include "sz3c.h"
#include <blosc2.h>

size_t compressElemBlosc2(const double* data, size_t nPoints,
                           std::vector<unsigned char>& out)
{
    size_t inputBytes = nPoints * sizeof(double);
    size_t maxOutput = inputBytes + BLOSC2_MAX_OVERHEAD;
    out.resize(maxOutput);

    blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS;
    cparams.compcode  = BLOSC_ZSTD;      // backend
    cparams.clevel    = 9;               // compression level
    cparams.typesize  = sizeof(double);
    cparams.nthreads  = 4;               // threads
    cparams.blocksize = 0;               // auto blocksize
    // cparams.filters[0] = BLOSC_BITSHUFFLE;   // stronger decorrelation


    blosc2_context* cctx = blosc2_create_cctx(cparams);
    int cbytes = blosc2_compress_ctx(cctx, data, inputBytes, out.data(), maxOutput);
    blosc2_free_ctx(cctx);

    if (cbytes <= 0) {
        fprintf(stderr, "Blosc2 compression failed: %d\n", cbytes);
        exit(1);
    }
    out.resize(cbytes);
    return cbytes;
}

std::vector<double> decompressElemBlosc2(const unsigned char* comp, size_t cbytes, size_t nPoints)
{
    size_t outputBytes = nPoints * sizeof(double);
    std::vector<double> out(nPoints);

    int dbytes = blosc2_decompress(
        comp,                         // compressed data
        static_cast<int32_t>(cbytes), // compressed size
        out.data(),                    // destination buffer
        static_cast<int32_t>(outputBytes) // uncompressed buffer size
    );

    if (dbytes <= 0) {
        fprintf(stderr, "Blosc2 decompression failed: %d\n", dbytes);
        exit(1);
    }

    return out;
}

struct CompressedField {
  std::string name;

  // concatenated compressed bytes for this rank (elem0 | elem1 | ...)
  std::vector<unsigned char> bytes;

  // per-element compressed sizes (one entry per element this rank owns)
  std::vector<uint64_t> blockSizes;

  // per-element offsets (local offsets inside bytes). optional, can be derived from blockSizes
  std::vector<uint64_t> blockOffsets;

  // Compression flag (0 = raw, 1 = compressed)
  uint8_t compressionFlag;

  // number of elements compressed in this rank
  uint64_t numElems = 0;

  // number of points per element (or total points on rank if rank-based)
  uint64_t ptsPerElem = 0;

  // Per-rank metadata (filled by reader)
  std::vector<uint64_t> perRankNumElems;      // size = nRanks
  std::vector<uint64_t> perRankPtsPerElem;    // size = nRanks
  std::vector<uint8_t>  perRankCompression;   // size = nRanks

  uint64_t originalSize = 0;   // total uncompressed bytes (local rank)
  uint64_t compressedSize = 0; // total compressed bytes (local rank)
  double maxError = 0.0;       // local max abs error
};

void writeCompressedFieldMPI(const std::string& filename,
                             const std::vector<CompressedField>& fields,
                             MPI_Comm comm)
{
  int rank, nRanks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nRanks);

  int numFields = (int)fields.size();

  // --- Step 1: compute per-field local meta/data sizes and global totals
  std::vector<uint64_t> myMetaSizes(numFields), totalMetaBytes(numFields);
  std::vector<uint64_t> myDataSizes(numFields), totalDataBytes(numFields);

  for (int f = 0; f < numFields; ++f) {
    myMetaSizes[f] = (uint64_t)fields[f].blockSizes.size() * sizeof(uint64_t); // bytes used to store blockSizes
    myDataSizes[f] = (uint64_t)fields[f].bytes.size();
  }
  // global totals per field (total metadata bytes across ranks, total data bytes across ranks)
  for (int f = 0; f < numFields; ++f) {
    MPI_Allreduce(&myMetaSizes[f], &totalMetaBytes[f], 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(&myDataSizes[f], &totalDataBytes[f], 1, MPI_UINT64_T, MPI_SUM, comm);
  }

  // --- Step 1b: prepare per-field per-rank local scalar values that must be gathered
  // We'll gather numElems, ptsPerElem, compressionFlag for each field to rank 0.
  // To gather, we use MPI_Gather per field.
  std::vector<uint64_t> gatheredNumElems;    // rank0 only: length = numFields * nRanks
  std::vector<uint64_t> gatheredPtsPerElem;  // rank0 only
  std::vector<uint8_t>  gatheredCompFlag;    // rank0 only

  if (rank == 0) {
    gatheredNumElems.resize((size_t)numFields * nRanks);
    gatheredPtsPerElem.resize((size_t)numFields * nRanks);
    gatheredCompFlag.resize((size_t)numFields * nRanks);
  }

  // We'll perform per-field MPI_Gather for each scalar on all ranks to rank 0
  for (int f = 0; f < numFields; ++f) {
    uint64_t localNumElems = fields[f].numElems;
    uint64_t localPts = fields[f].ptsPerElem;
    uint8_t  localFlag = fields[f].compressionFlag;

    // collect numeric arrays to rank 0
    MPI_Gather(&localNumElems, 1, MPI_UINT64_T,
               rank == 0 ? &gatheredNumElems[(size_t)f * nRanks] : nullptr,
               1, MPI_UINT64_T, 0, comm);

    MPI_Gather(&localPts, 1, MPI_UINT64_T,
               rank == 0 ? &gatheredPtsPerElem[(size_t)f * nRanks] : nullptr,
               1, MPI_UINT64_T, 0, comm);

    MPI_Gather(&localFlag, 1, MPI_UNSIGNED_CHAR,
               rank == 0 ? &gatheredCompFlag[(size_t)f * nRanks] : nullptr,
               1, MPI_UNSIGNED_CHAR, 0, comm);
  }

  // --- Step 2: compute header size and per-field metadata/payload offsets (rank 0 computes, then broadcast)
  // Header layout:
  // [int numFields]
  // For each field f:
  //   [int nameLen] [name bytes]
  //   [uint64_t totalMetaBytes] [uint64_t totalDataBytes]
  //   [uint64_t nRanks]
  //   for r=0..nRanks-1: [uint64_t numElems[r]] [uint64_t ptsPerElem[r]] [uint8_t compressionFlag[r]]
  //   [uint64_t metaRegionOffset] [uint64_t dataRegionOffset]
  std::vector<uint64_t> fieldMetaOffset(numFields), fieldDataOffset(numFields);

  MPI_Offset headerSize = 0;
  if (rank == 0) {
    headerSize = sizeof(int); // numFields
    for (int f = 0; f < numFields; ++f) {
      headerSize += sizeof(int); // nameLen
      headerSize += fields[f].name.size(); // name bytes
      headerSize += sizeof(uint64_t); // totalMetaBytes
      headerSize += sizeof(uint64_t); // totalDataBytes
      headerSize += sizeof(uint64_t); // nRanks
      // per-rank meta: for each rank we store numElems(uint64_t), ptsPerElem(uint64_t), compFlag(uint8_t)
      headerSize += (MPI_Offset)nRanks * (sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint8_t));
      // padding/alignment is not added; you could align to 8 if desired
      headerSize += sizeof(uint64_t); // metaRegionOffset
      headerSize += sizeof(uint64_t); // dataRegionOffset
    }

    // compute offsets for each field's metadata and data regions (meta = blockSizes region)
    MPI_Offset cursor = headerSize;
    for (int f = 0; f < numFields; ++f) {
      fieldMetaOffset[f] = (uint64_t)cursor;
      cursor += (MPI_Offset)totalMetaBytes[f];
    }
    for (int f = 0; f < numFields; ++f) {
      fieldDataOffset[f] = (uint64_t)cursor;
      cursor += (MPI_Offset)totalDataBytes[f];
    }
  }

  // broadcast headerSize and region offsets to all ranks
  MPI_Bcast(&headerSize, 1, MPI_LONG_LONG_INT, 0, comm); // MPI_Offset may map to long long
  MPI_Bcast(fieldMetaOffset.data(), numFields, MPI_UINT64_T, 0, comm);
  MPI_Bcast(fieldDataOffset.data(), numFields, MPI_UINT64_T, 0, comm);

  // --- Step 3: open file collectively
  MPI_File fh;
  MPI_File_open(comm, filename.c_str(),
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);

  // --- Step 4: rank 0 writes header (names + totals + per-rank meta + region offsets)
  if (rank == 0) {
    MPI_Offset pos = 0;
    MPI_File_write_at(fh, pos, &numFields, 1, MPI_INT, MPI_STATUS_IGNORE);
    pos += sizeof(int);

    for (int f = 0; f < numFields; ++f) {
      int nameLen = (int)fields[f].name.size();
      MPI_File_write_at(fh, pos, &nameLen, 1, MPI_INT, MPI_STATUS_IGNORE);
      pos += sizeof(int);

      MPI_File_write_at(fh, pos, fields[f].name.data(), nameLen, MPI_CHAR, MPI_STATUS_IGNORE);
      pos += nameLen;

      // write totals (global)
      MPI_File_write_at(fh, pos, &totalMetaBytes[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE);
      pos += sizeof(uint64_t);
      MPI_File_write_at(fh, pos, &totalDataBytes[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE);
      pos += sizeof(uint64_t);

      // write nRanks (as uint64)
      uint64_t nr = (uint64_t)nRanks;
      MPI_File_write_at(fh, pos, &nr, 1, MPI_UINT64_T, MPI_STATUS_IGNORE);
      pos += sizeof(uint64_t);

      // write per-rank metadata gathered into gatheredNumElems/gatheredPtsPerElem/gatheredCompFlag
      // they are stored in interleaved blocks: for field f, the slice is gatheredNumElems[f*nRanks + r]
      for (int r = 0; r < nRanks; ++r) {
        uint64_t ne = gatheredNumElems[(size_t)f * nRanks + r];
        uint64_t pts = gatheredPtsPerElem[(size_t)f * nRanks + r];
        uint8_t  fl = gatheredCompFlag[(size_t)f * nRanks + r];

        MPI_File_write_at(fh, pos, &ne, 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
        MPI_File_write_at(fh, pos, &pts, 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
        MPI_File_write_at(fh, pos, &fl, 1, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE); pos += sizeof(uint8_t);
      }

      // write the region offsets
      uint64_t metaOff = fieldMetaOffset[f];
      uint64_t dataOff = fieldDataOffset[f];
      MPI_File_write_at(fh, pos, &metaOff, 1, MPI_UINT64_T, MPI_STATUS_IGNORE);
      pos += sizeof(uint64_t);
      MPI_File_write_at(fh, pos, &dataOff, 1, MPI_UINT64_T, MPI_STATUS_IGNORE);
      pos += sizeof(uint64_t);
    }
  }

  MPI_Barrier(comm); // ensure header is written before metadata/payload

  // --- Step 5: each rank writes its per-field metadata (blockSizes) in parallel
  for (int f = 0; f < numFields; ++f) {
    // compute prefix to know where *this rank's* metadata starts within the field metadata region
    uint64_t myMeta = myMetaSizes[f];
    uint64_t myMetaDispl = 0;
    MPI_Exscan(&myMeta, &myMetaDispl, 1, MPI_UINT64_T, MPI_SUM, comm);
    if (rank == 0) myMetaDispl = 0;

    MPI_Offset writePos = (MPI_Offset)fieldMetaOffset[f] + (MPI_Offset)myMetaDispl;

    if (!fields[f].blockSizes.empty()) {
      // write blockSizes (each as uint64_t)
      MPI_File_write_at_all(fh, writePos,
                            fields[f].blockSizes.data(),
                            fields[f].blockSizes.size(),
                            MPI_UINT64_T, MPI_STATUS_IGNORE);
    }
  }

  MPI_Barrier(comm);

  // --- Step 6: each rank writes its per-field payload (concatenated bytes) in parallel
  for (int f = 0; f < numFields; ++f) {
    uint64_t myData = myDataSizes[f];
    uint64_t myDataDispl = 0;
    MPI_Exscan(&myData, &myDataDispl, 1, MPI_UINT64_T, MPI_SUM, comm);
    if (rank == 0) myDataDispl = 0;

    MPI_Offset writePos = (MPI_Offset)fieldDataOffset[f] + (MPI_Offset)myDataDispl;

    if (!fields[f].bytes.empty()) {
      MPI_File_write_at_all(fh, writePos,
                            fields[f].bytes.data(),
                            fields[f].bytes.size(),
                            MPI_BYTE, MPI_STATUS_IGNORE);
    }
  }

  MPI_File_close(&fh);
}

template <typename T = dfloat>
void compressAndReport(nrs_t* nrs, double relEb, occa::memory &o_V, occa::memory &o_Ve, const std::string& fname) {
  mesh_t* mesh = nrs->mesh;

  // Prepare host buffers (as before)
  std::vector<T> X(mesh->Nlocal);
  std::vector<T> Y(mesh->Nlocal);
  std::vector<T> Z(mesh->Nlocal);

  std::vector<T> Ux(mesh->Nlocal);
  std::vector<T> Uy(mesh->Nlocal);
  std::vector<T> Uz(mesh->Nlocal);

  (mesh->o_x).copyTo(X.data(), mesh->Nlocal);
  (mesh->o_y).copyTo(Y.data(), mesh->Nlocal);
  (mesh->o_z).copyTo(Z.data(), mesh->Nlocal);
  
  (nrs->o_U + 0 * nrs->fieldOffset).copyTo(Ux.data(), mesh->Nlocal);
  (nrs->o_U + 1 * nrs->fieldOffset).copyTo(Uy.data(), mesh->Nlocal);
  (nrs->o_U + 2 * nrs->fieldOffset).copyTo(Uz.data(), mesh->Nlocal);

  const int Np = mesh->N + 1;
  const int ptsPerElem = mesh->Nelements * Np * Np * Np;
  const int Ne = 1; // number of elements on this rank

  // processField now returns CompressedField
  auto processField = [&](const std::vector<T>& data,
                          const std::string& name,
                          double tol,
                          occa::memory U,
                          occa::memory Ue) -> CompressedField
  {
    CompressedField f;
    f.name = name;
    f.numElems   = Ne;          // 1 in rank-based mode
    f.ptsPerElem = ptsPerElem;  // total points per rank in rank-based mode
    if(tol>0) f.compressionFlag = 1;
    else f.compressionFlag = 0;

    // buffers for recovered values and error (full field)
    std::vector<T> recoveredAll(mesh->Nlocal);
    std::vector<T> errorAll(mesh->Nlocal);

    double localMaxErr = 0.0;
    f.originalSize = 0;
    f.compressedSize = 0;

    // iterate elements on this rank
    f.blockSizes.reserve(Ne);
    f.blockOffsets.reserve(Ne);

    if(f.compressionFlag==0){
      for (int e = 0; e < Ne; ++e) {
        const T* elemData = data.data() + (size_t)e * ptsPerElem;

        // --- Allocate output buffer for raw copy ---
        size_t elemBytes = (size_t)ptsPerElem * sizeof(T);
        unsigned char* compressed = (unsigned char*) malloc(elemBytes);

        // --- Copy raw bytes ---
        memcpy(compressed, elemData, elemBytes);

        // Byte size of this “compressed” block
        size_t outSize = elemBytes;

        // --- Store block metadata ---
        uint64_t curOffset = (uint64_t) f.bytes.size();
        f.blockOffsets.push_back(curOffset);
        f.blockSizes.push_back((uint64_t) outSize);

        f.bytes.insert(
            f.bytes.end(),
            compressed,
            compressed + outSize
        );

        f.originalSize   += elemBytes;
        f.compressedSize += outSize;

        // For uncompressed case, recovered == original
        const T* recovered = elemData;

        for (int i = 0; i < ptsPerElem; ++i) {
          size_t gid = (size_t)e * ptsPerElem + i;
          recoveredAll[gid] = recovered[i];

          double err = data[gid] - recovered[i]; // always 0 here
          errorAll[gid] = (T)err;

          if (std::abs(err) > localMaxErr)
            localMaxErr = std::abs(err);
        }

        free(compressed);
      }
    }else {
      for (int e = 0; e < Ne; ++e) {
        const T* elemData = data.data() + (size_t)e * ptsPerElem;

        size_t outSize = 0;
        unsigned char* compressed = SZ_compress_args(
          std::is_same<T,double>::value ? SZ_DOUBLE : SZ_FLOAT,
          (void*)elemData,
          &outSize,
          ABS,
          tol,
          0,
          0,
          0,0,0,1,ptsPerElem
        );

        // decompress to compute errors and recovered values
        T* recovered = (T*) SZ_decompress(
          std::is_same<T,double>::value ? SZ_DOUBLE : SZ_FLOAT,
          compressed,
          outSize,
          0,0,0,1,ptsPerElem
        );

        // std::vector<unsigned char> compressedVec;
        // size_t outSize = compressElemBlosc2(elemData, ptsPerElem, compressedVec);
        // unsigned char* compressed = compressedVec.data(); // for compatibility with rest of code
        
        // std::vector<T> recoveredVec = decompressElemBlosc2(compressed, outSize, ptsPerElem);
        // T* recovered = recoveredVec.data();

        // compute error and copy to recoveredAll/errorAll
        for (int i = 0; i < ptsPerElem; ++i) {
          size_t gid = (size_t)e * ptsPerElem + i;
          recoveredAll[gid] = recovered[i];
          double err = (double) (data[gid] - recovered[i]);
          errorAll[gid] = (T) err;
          if (std::abs(err) > localMaxErr) localMaxErr = std::abs(err);
        }

        // append compressed block into bytes and record blockSizes/offsets
        uint64_t curOffset = (uint64_t) f.bytes.size();
        f.blockOffsets.push_back(curOffset);
        f.blockSizes.push_back((uint64_t) outSize);
        f.bytes.insert(f.bytes.end(), compressed, compressed + outSize);

        f.originalSize += (uint64_t) (ptsPerElem * sizeof(T));
        f.compressedSize += (uint64_t) outSize;

        free_buf(compressed);
        free_buf(recovered);
      }
    }
 
    // compute global reductions (totals across ranks) for printing
    uint64_t localOrig = f.originalSize;
    uint64_t localCmp = f.compressedSize;
    uint64_t totalOrig = 0, totalCmp = 0;
    MPI_Allreduce(&localOrig, &totalOrig, 1, MPI_UINT64_T, MPI_SUM, platform->comm.mpiComm);
    MPI_Allreduce(&localCmp, &totalCmp, 1, MPI_UINT64_T, MPI_SUM, platform->comm.mpiComm);

    double globalMaxErr = localMaxErr;
    MPI_Allreduce(MPI_IN_PLACE, &globalMaxErr, 1,
                  std::is_same<T,double>::value ? MPI_DOUBLE : MPI_DOUBLE, // double is safe for reporting
                  MPI_MAX, platform->comm.mpiComm);

    if (platform->comm.mpiRank == 0) {
      printf("%s (all ranks combined, tol=%g):\n", name.c_str(), tol);
      printf("  Total original size    = %llu bytes\n", (unsigned long long) totalOrig);
      printf("  Total compressed size  = %llu bytes\n", (unsigned long long) totalCmp);
      printf("  Compression ratio      = %.3f\n",
            (double)totalOrig / (double)totalCmp);
      printf("  Max abs error          = %g\n", globalMaxErr);
    }

    // copy recovered and error back to device views
    U.copyFrom(recoveredAll.data(), mesh->Nlocal);
    Ue.copyFrom(errorAll.data(), mesh->Nlocal);

    f.maxError = localMaxErr;
    return f;

  }; // end processField

  // call processField for each field and collect results
  std::vector<CompressedField> fields;
  fields.reserve(6);
  fields.push_back(processField(X , "X" , 0.0*relEb , o_V + 0*nrs->fieldOffset, o_Ve + 0*nrs->fieldOffset)); // o_Ve is placeholder
  fields.push_back(processField(Y , "Y" , 0.0*relEb , o_V + 0*nrs->fieldOffset, o_Ve + 0*nrs->fieldOffset)); // o_Ve is placeholder
  fields.push_back(processField(Z , "Z" , 0.0*relEb , o_V + 0*nrs->fieldOffset, o_Ve + 0*nrs->fieldOffset)); // o_Ve is placeholder

  fields.push_back(processField(Ux, "Ux", 1.0*relEb , o_V + 0*nrs->fieldOffset, o_Ve + 0*nrs->fieldOffset)); // overwrite o_Ve
  fields.push_back(processField(Uy, "Uy", 1.0*relEb , o_V + 1*nrs->fieldOffset, o_Ve + 1*nrs->fieldOffset));
  fields.push_back(processField(Uz, "Uz", 1.0*relEb , o_V + 2*nrs->fieldOffset, o_Ve + 2*nrs->fieldOffset));
  // push additional scalar fields as needed

  // write all fields simultaneously to a single file (parallel)
  writeCompressedFieldMPI(fname, fields, platform->comm.mpiComm);
}


std::vector<CompressedField> readCompressedFieldMPI(const std::string &filename, MPI_Comm comm)
{
    int rank, nRanks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nRanks);

    MPI_File fh;
    if (MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "ERROR opening file %s\n", filename.c_str());
        MPI_Abort(comm, 1);
    }

    int numFields = 0;
    if (rank == 0) {
        MPI_File_read_at(fh, 0, &numFields, 1, MPI_INT, MPI_STATUS_IGNORE);
    }
    MPI_Bcast(&numFields, 1, MPI_INT, 0, comm);

    std::vector<std::string> fieldNames(numFields);
    std::vector<uint64_t> totalMetaBytes(numFields), totalDataBytes(numFields);
    std::vector<uint64_t> fieldMetaOffset(numFields), fieldDataOffset(numFields);
    std::vector<std::vector<uint64_t>> perFieldNumElems(numFields);
    std::vector<std::vector<uint64_t>> perFieldPtsPerElem(numFields);
    std::vector<std::vector<uint8_t>>  perFieldCompFlag(numFields);

    MPI_Offset pos = sizeof(int); // after numFields

    if (rank == 0) {
        for (int f = 0; f < numFields; ++f) {
            int nameLen = 0;
            MPI_File_read_at(fh, pos, &nameLen, 1, MPI_INT, MPI_STATUS_IGNORE);
            pos += sizeof(int);

            fieldNames[f].resize(nameLen);
            if (nameLen > 0)
                MPI_File_read_at(fh, pos, &fieldNames[f][0], nameLen, MPI_CHAR, MPI_STATUS_IGNORE);
            pos += nameLen;

            MPI_File_read_at(fh, pos, &totalMetaBytes[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += 8;
            MPI_File_read_at(fh, pos, &totalDataBytes[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += 8;

            uint64_t nRF = 0;
            MPI_File_read_at(fh, pos, &nRF, 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += 8;

            perFieldNumElems[f].resize(nRF);
            perFieldPtsPerElem[f].resize(nRF);
            perFieldCompFlag[f].resize(nRF);

            for (uint64_t r = 0; r < nRF; ++r) {
                MPI_File_read_at(fh, pos, &perFieldNumElems[f][r], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += 8;
                MPI_File_read_at(fh, pos, &perFieldPtsPerElem[f][r], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += 8;
                MPI_File_read_at(fh, pos, &perFieldCompFlag[f][r], 1, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE); pos += 1;
            }

            MPI_File_read_at(fh, pos, &fieldMetaOffset[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += 8;
            MPI_File_read_at(fh, pos, &fieldDataOffset[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += 8;
        }
    }

    // Broadcast all header info to all ranks
    for (int f = 0; f < numFields; ++f) {
        int nameLen = (rank == 0 ? (int)fieldNames[f].size() : 0);
        MPI_Bcast(&nameLen, 1, MPI_INT, 0, comm);

        fieldNames[f].resize(nameLen);
        if (nameLen > 0) MPI_Bcast(&fieldNames[f][0], nameLen, MPI_CHAR, 0, comm);

        MPI_Bcast(&totalMetaBytes[f], 1, MPI_UINT64_T, 0, comm);
        MPI_Bcast(&totalDataBytes[f], 1, MPI_UINT64_T, 0, comm);
        MPI_Bcast(&fieldMetaOffset[f], 1, MPI_UINT64_T, 0, comm);
        MPI_Bcast(&fieldDataOffset[f], 1, MPI_UINT64_T, 0, comm);

        uint64_t nRF = (rank == 0 ? perFieldNumElems[f].size() : 0);
        MPI_Bcast(&nRF, 1, MPI_UINT64_T, 0, comm);

        perFieldNumElems[f].resize(nRF);
        perFieldPtsPerElem[f].resize(nRF);
        perFieldCompFlag[f].resize(nRF);

        MPI_Bcast(perFieldNumElems[f].data(), nRF, MPI_UINT64_T, 0, comm);
        MPI_Bcast(perFieldPtsPerElem[f].data(), nRF, MPI_UINT64_T, 0, comm);
        MPI_Bcast(perFieldCompFlag[f].data(), nRF, MPI_UNSIGNED_CHAR, 0, comm);
    }

    std::vector<CompressedField> out(numFields);

    // --- read blockSizes metadata
    for (int f = 0; f < numFields; ++f) {
        auto &CF = out[f];
        CF.name = fieldNames[f];
        CF.perRankNumElems = perFieldNumElems[f];
        CF.perRankPtsPerElem = perFieldPtsPerElem[f];
        CF.perRankCompression = perFieldCompFlag[f];

        size_t myBlocks = (rank < (int)CF.perRankNumElems.size() ? CF.perRankNumElems[rank] : 0);
        uint64_t myMetaBytes = myBlocks * sizeof(uint64_t);
        uint64_t myMetaDispl = 0;
        MPI_Exscan(&myMetaBytes, &myMetaDispl, 1, MPI_UINT64_T, MPI_SUM, comm);
        if (rank == 0) myMetaDispl = 0;

        CF.blockSizes.resize(myBlocks);
        if (myBlocks)
            MPI_File_read_at_all(fh, fieldMetaOffset[f] + myMetaDispl,
                                 CF.blockSizes.data(), myBlocks, MPI_UINT64_T,
                                 MPI_STATUS_IGNORE);

        CF.blockOffsets.resize(myBlocks);
        uint64_t offset = 0;
        for (size_t b = 0; b < myBlocks; ++b) {
            CF.blockOffsets[b] = offset;
            offset += CF.blockSizes[b];
        }
        CF.compressedSize = offset;

        CF.numElems   = myBlocks;
        CF.ptsPerElem = (rank < (int)CF.perRankPtsPerElem.size() ? CF.perRankPtsPerElem[rank] : 0);
        CF.compressionFlag = (rank < (int)CF.perRankCompression.size() ? CF.perRankCompression[rank] : 0);
        CF.originalSize = CF.numElems * CF.ptsPerElem * sizeof(double);
    }

    // --- read payload using pure MPI ---
    for (int f = 0; f < numFields; ++f) {
        auto &CF = out[f];
        uint64_t myData = CF.compressedSize;
        if (myData == 0) continue;

        uint64_t myDispl = 0;
        MPI_Exscan(&myData, &myDispl, 1, MPI_UINT64_T, MPI_SUM, comm);
        if (rank == 0) myDispl = 0;

        CF.bytes.resize(myData);
        MPI_File_read_at_all(fh, fieldDataOffset[f] + myDispl,
                             CF.bytes.data(), myData, MPI_BYTE, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fh);
    return out;
}

template <typename T = dfloat>
std::vector<T> decompressField(nrs_t* nrs, const CompressedField &f, bool debug=true)
{
    mesh_t* mesh = nrs->mesh;
    uint64_t ptsPerElem = f.ptsPerElem;
    uint64_t numElems   = f.numElems;
    uint8_t compressionFlag = f.compressionFlag;

    if (ptsPerElem==0 || numElems==0) {
        if (debug || platform->comm.mpiRank==0)
            fprintf(stderr,"ERROR decompressField: zero ptsPerElem=%llu or numElems=%llu for '%s'\n",
                    (unsigned long long)ptsPerElem, (unsigned long long)numElems, f.name.c_str());
        return {};
    }

    std::vector<T> recovered((size_t)numElems * (size_t)ptsPerElem);

    for (size_t e = 0; e < numElems; ++e) {
        uint64_t offset = (e<f.blockOffsets.size()?f.blockOffsets[e]:0);
        uint64_t bsize  = (e<f.blockSizes.size()?f.blockSizes[e]:0);

        if (offset + bsize > f.bytes.size()) {
            if (debug || platform->comm.mpiRank==0)
                fprintf(stderr,"ERROR decompressField('%s'): block %zu offset+size exceeds bytes.size\n",
                        f.name.c_str(), e);
            return {};
        }

        const unsigned char* compPtr = f.bytes.data() + offset;

        if (compressionFlag==0) {
            if (bsize % sizeof(T) != 0) {
                if (debug || platform->comm.mpiRank==0)
                    fprintf(stderr,"ERROR decompressField('%s'): raw block %zu size %llu not divisible by sizeof(T)\n",
                            f.name.c_str(), e, (unsigned long long)bsize);
                return {};
            }
            size_t nT = bsize / sizeof(T);
            const T* src = reinterpret_cast<const T*>(compPtr);
            for (size_t i=0;i<std::min(nT,(size_t)ptsPerElem);++i)
                recovered[e*ptsPerElem + i] = src[i];
            for (size_t i=nT;i<ptsPerElem;++i)
                recovered[e*ptsPerElem + i] = (T)0;
        } else {
            void* dec = SZ_decompress(std::is_same<T,double>::value?SZ_DOUBLE:SZ_FLOAT,
                                      const_cast<unsigned char*>(compPtr), (size_t)bsize,
                                      0,0,0,1,(size_t)ptsPerElem);
            if (!dec) {
                if (debug || platform->comm.mpiRank==0)
                    fprintf(stderr,"ERROR decompressField('%s'): SZ_decompress returned NULL for block %zu\n",
                            f.name.c_str(), e);
                return {};
            }
            T* elemRec = reinterpret_cast<T*>(dec);
            for (size_t i=0;i<ptsPerElem;++i)
                recovered[e*ptsPerElem + i] = elemRec[i];
            free_buf(dec);

            // // Blosc only
            // size_t outBytes = ptsPerElem*sizeof(T);
            // int rc = blosc2_decompress(compPtr, recovered.data()+e*ptsPerElem, outBytes);
            // if (rc <= 0) {
            //     if (debug) fprintf(stderr,"ERROR decompressFieldHybrid: Blosc failed for block %zu\n", e);
            //     return {};
            // }
        }
    }

    if (debug || platform->comm.mpiRank==0) {
        size_t nprint = std::min((size_t)8, recovered.size());
        fprintf(stderr,"decompressField('%s') -> totalPts=%zu sample:", f.name.c_str(), recovered.size());
        for (size_t i=0;i<nprint;++i) fprintf(stderr," %g",(double)recovered[i]);
        fprintf(stderr,"\n");
    }

    return recovered;
}

void outfld_wrapper(nrs_t *nrs, std::unique_ptr<iofld> &checkpointWriter, const int N, occa::memory U, double time, int tstep, std::string fileName)
{ 
  if (!checkpointWriter) {
    checkpointWriter = iofldFactory::create("nek"); // or "adios"
    if (platform->comm.mpiRank == 0) {
      printf("create a new iofldFactory... %s\n", fileName.c_str());
    }
  }
  
  if (!checkpointWriter->isInitialized()) {
    auto visMesh = (nrs->cht) ? nrs->cds->mesh[0] : nrs->mesh;
    checkpointWriter->open(visMesh, iofld::mode::write, fileName);
    
    if (platform->options.compareArgs("LOWMACH", "TRUE")) {
      checkpointWriter->addVariable("p0th", nrs->p0th[0]);
    }
    
    if (platform->options.compareArgs("VELOCITY CHECKPOINTING", "TRUE")) {
      std::vector<occa::memory> o_V;
      for (int i = 0; i < visMesh->dim; i++) {
        o_V.push_back(U.slice(i * nrs->fieldOffset, visMesh->Nlocal));
      }
      checkpointWriter->addVariable("velocity", o_V);
    }

    if (platform->options.compareArgs("PRESSURE CHECKPOINTING", "TRUE")) {
      auto o_p = std::vector<occa::memory>{nrs->o_P.slice(0, visMesh->Nlocal)};
      checkpointWriter->addVariable("pressure", o_p);
    }

    for (int i = 0; i < nrs->Nscalar; i++) {
      if (platform->options.compareArgs("SCALAR" + scalarDigitStr(i) + " CHECKPOINTING", "TRUE")) {
        const auto temperatureExists = platform->options.compareArgs("SCALAR00 IS TEMPERATURE", "TRUE");
        std::vector<occa::memory> o_Si = {nrs->cds->o_S.slice(nrs->cds->fieldOffsetScan[i], visMesh->Nlocal)};
        if (i == 0 && temperatureExists) {
          checkpointWriter->addVariable("temperature", o_Si);
        } else {
          const auto is = (temperatureExists) ? i - 1 : i;
          checkpointWriter->addVariable("scalar" + scalarDigitStr(is), o_Si);
        }
      }
    }
    
  }
  
  const auto outXYZ = platform->options.compareArgs("CHECKPOINT OUTPUT MESH", "TRUE");
  const auto FP64 = platform->options.compareArgs("CHECKPOINT PRECISION", "FP64");
  const auto uniform = (N < 0) ? true : false;
  
  checkpointWriter->writeAttribute("polynomialOrder", std::to_string(abs(N)));
  checkpointWriter->writeAttribute("precision", (FP64) ? "64" : "32");
  checkpointWriter->writeAttribute("uniform", (uniform) ? "true" : "false");
  checkpointWriter->writeAttribute("outputMesh", (outXYZ) ? "true" : "false");

  checkpointWriter->addVariable("time", time);
  
  checkpointWriter->process();
}
