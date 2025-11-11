#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdio.h>   
#include <stdlib.h>   

#include "sz3c.h"
#include <mpi.h>

struct CompressedField {
  std::string name;

  // concatenated compressed bytes for this rank (elem0 | elem1 | ...)
  std::vector<unsigned char> bytes;

  // per-element compressed sizes (one entry per element this rank owns)
  std::vector<uint64_t> blockSizes;

  // per-element offsets (local offsets inside bytes). optional, can be derived from blockSizes
  std::vector<uint64_t> blockOffsets;

  uint64_t originalSize = 0;   // total uncompressed bytes (local rank)
  uint64_t compressedSize = 0; // total compressed bytes (local rank)
  double maxError = 0.0;       // local max abs error
};

std::vector<CompressedField> readCompressedFieldMPI(const std::string &filename, MPI_Comm comm)
{
    int rank, nRanks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nRanks);

    MPI_File fh;
    MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    int numFields = 0;
    MPI_Offset pos = 0;

    // --- Rank 0 reads header ---
    std::vector<std::string> fieldNames;
    std::vector<uint64_t> totalMetaBytes, totalDataBytes;
    std::vector<uint64_t> fieldMetaOffset, fieldDataOffset;

    if (rank == 0) {
        MPI_File_read_at(fh, pos, &numFields, 1, MPI_INT, MPI_STATUS_IGNORE);
        std::cout<<"numFields: "<<numFields<<std::endl; //
        pos += sizeof(int);

        fieldNames.resize(numFields);
        totalMetaBytes.resize(numFields);
        totalDataBytes.resize(numFields);
        fieldMetaOffset.resize(numFields);
        fieldDataOffset.resize(numFields);

        for (int f = 0; f < numFields; ++f) {
            int nameLen = 0;
            MPI_File_read_at(fh, pos, &nameLen, 1, MPI_INT, MPI_STATUS_IGNORE);
            std::cout<<"nameLen: "<<nameLen<<std::endl; //
            pos += sizeof(int);

            fieldNames[f].resize(nameLen);
            MPI_File_read_at(fh, pos, fieldNames[f].data(), nameLen, MPI_CHAR, MPI_STATUS_IGNORE);
            std::cout<<"fieldName: "<<fieldNames[f]<<std::endl; //
            pos += nameLen;

            MPI_File_read_at(fh, pos, &totalMetaBytes[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
            MPI_File_read_at(fh, pos, &totalDataBytes[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
            MPI_File_read_at(fh, pos, &fieldMetaOffset[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
            MPI_File_read_at(fh, pos, &fieldDataOffset[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);

            std::cout<<"totalMetaBytes: "<<totalMetaBytes[f]<<std::endl; //
            std::cout<<"totalDataBytes: "<<totalDataBytes[f]<<std::endl; //
            std::cout<<"fieldMetaOffset: "<<fieldMetaOffset[f]<<std::endl; //
            std::cout<<"fieldDataOffset: "<<fieldDataOffset[f]<<std::endl; //
        }
    }

    // broadcast to all ranks
    MPI_Bcast(&numFields, 1, MPI_INT, 0, comm);
    if (rank != 0) {
        fieldNames.resize(numFields);
        totalMetaBytes.resize(numFields);
        totalDataBytes.resize(numFields);
        fieldMetaOffset.resize(numFields);
        fieldDataOffset.resize(numFields);
    }
    for (int f=0; f<numFields; ++f) {
        int nameLen = fieldNames[f].size();
        MPI_Bcast(&nameLen, 1, MPI_INT, 0, comm);
        if (rank != 0) fieldNames[f].resize(nameLen);
        MPI_Bcast(fieldNames[f].data(), nameLen, MPI_CHAR, 0, comm);

        MPI_Bcast(&totalMetaBytes[f], 1, MPI_UINT64_T, 0, comm);
        MPI_Bcast(&totalDataBytes[f], 1, MPI_UINT64_T, 0, comm);
        MPI_Bcast(&fieldMetaOffset[f], 1, MPI_UINT64_T, 0, comm);
        MPI_Bcast(&fieldDataOffset[f], 1, MPI_UINT64_T, 0, comm);
    }

    // --- Read per-field metadata (blockSizes) ---
    std::vector<CompressedField> fields(numFields);

    for (int f=0; f<numFields; ++f) {
        fields[f].name = fieldNames[f];

        // determine my metadata size for this rank
        uint64_t myMeta = totalMetaBytes[f] / nRanks; // approx
        if(myMeta % 8 != 0) myMeta = (myMeta / 8) * 8;
        if(rank == nRanks - 1) myMeta = totalMetaBytes[f] - myMeta * (nRanks - 1);
        assert(myMeta % 8 == 0);

        uint64_t myMetaDispl = 0;
        MPI_Exscan(&myMeta, &myMetaDispl, 1, MPI_UINT64_T, MPI_SUM, comm);
        if (rank == 0) myMetaDispl = 0;

        size_t numBlocks = myMeta / sizeof(uint64_t);
        fields[f].blockSizes.resize(numBlocks);
        MPI_File_read_at_all(fh, fieldMetaOffset[f] + myMetaDispl,
                             fields[f].blockSizes.data(), numBlocks, MPI_UINT64_T, MPI_STATUS_IGNORE);

        // compute block offsets
        fields[f].blockOffsets.resize(numBlocks);
        size_t offset = 0;
        for (size_t b = 0; b < numBlocks; ++b) {
            fields[f].blockOffsets[b] = offset;
            offset += fields[f].blockSizes[b];
        }

        // total sizes for this rank
        fields[f].compressedSize = offset;
        fields[f].originalSize = 0; // user can store separately or compute from Np^3*numBlocks
    }

    // --- Read per-field payload ---
    for (int f=0; f<numFields; ++f) {
        uint64_t myData = fields[f].compressedSize;
        uint64_t myDataDispl = 0;
        MPI_Exscan(&myData, &myDataDispl, 1, MPI_UINT64_T, MPI_SUM, comm);
        if (rank == 0) myDataDispl = 0;

        fields[f].bytes.resize(fields[f].compressedSize);
        MPI_File_read_at_all(fh, fieldDataOffset[f] + myDataDispl,
                             fields[f].bytes.data(), fields[f].compressedSize, MPI_BYTE, MPI_STATUS_IGNORE);
    }

    MPI_File_close(&fh);

    // sanity check
    std::vector<uint64_t> allRanksDataBytes(numFields);
    for (int f=0; f<numFields; ++f) {
        uint64_t myData = fields[f].bytes.size();
        MPI_Allreduce(
        &myData,
        &allRanksDataBytes[f],    
        1,                  
        MPI_UINT64_T,            
        MPI_SUM,            
        comm);
        // Make sure all daya bytes are read.
        assert(allRanksDataBytes[f] == totalDataBytes[f]);
    }
    

    return fields;
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 2) {
        if (world_rank == 0) {
            std::cerr << "Usage: mpirun -n <N> " << argv[0] << " <filename.dat>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string filename = argv[1];

    if (world_rank == 0) {
        std::cout << "Reading compressed file: " << filename << std::endl;
    }

    std::vector<CompressedField> all_fields = readCompressedFieldMPI(filename, MPI_COMM_WORLD);


    if (world_rank == 0) {
        std::cout << "File read complete."<<std::endl;
    }

    MPI_Finalize();
    return 0;
}