#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdio.h>   
#include <stdlib.h>
#include <stdexcept>  
#include <cstring> 

#include "sz3c.h"
#include <mpi.h>

struct CompressedFieldPerRank {

    std::string name;

    // concatenated compressed bytes of compressed blocks (block0|block1|block2|...)
    std::vector<unsigned char> bytes;

    // number of points per block (one entry per block)
    std::vector<uint64_t> ptsPerBlock;

    // per-block compressed sizes in bytes (one entry per block)
    std::vector<uint64_t> compressedSizes;

    // per-block offsets derived from blockSizes (local rank offsets inside bytes, one entry per block) 
    std::vector<uint64_t> blockOffsets;

    // compression flags for each block (0 = raw, 1 = compressed, one entry per block)
    std::vector<uint8_t> compressionFlags;

    // number of compressed blocks in this rank
    uint64_t numBlocks = 0;

    // total compressed bytes (local rank)
    uint64_t compressedSize = 0;
};

std::vector<CompressedFieldPerRank> readCompressedFieldMPI(const std::string &filename, MPI_Comm comm)
{
    int rank, nRanks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nRanks);

    MPI_File fh;
    if (MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "ERROR opening file %s\n", filename.c_str());
        MPI_Abort(comm, 1);
    }
    
    MPI_Offset pos = 0;

    int numFields = 0;
    if (rank == 0) {
        MPI_File_read_at(fh, pos, &numFields, 1, MPI_INT, MPI_STATUS_IGNORE);
        std::cout<<"numFields: "<<numFields<<std::endl; //debug
    }
    MPI_Bcast(&numFields, 1, MPI_INT, 0, comm);
    pos += sizeof(int);

    std::vector<std::string> fieldNames(numFields);
    std::vector<uint64_t> totalMetaBytes(numFields), totalDataBytes(numFields);
    std::vector<uint64_t> fieldMetaOffset(numFields), fieldDataOffset(numFields);
    std::vector<std::vector<uint64_t>> perFieldNumElems(numFields);
    std::vector<std::vector<uint64_t>> perFieldPtsPerElem(numFields);
    std::vector<std::vector<uint8_t>>  perFieldCompFlag(numFields);

    if (rank == 0) {
        for (int f = 0; f < numFields; ++f) {
            int nameLen = 0;
            MPI_File_read_at(fh, pos, &nameLen, 1, MPI_INT, MPI_STATUS_IGNORE);
            pos += sizeof(int);
            std::cout<<"nameLen: "<<nameLen<<std::endl; //debug

            fieldNames[f].resize(nameLen);
            if (nameLen > 0)
                MPI_File_read_at(fh, pos, &fieldNames[f][0], nameLen, MPI_CHAR, MPI_STATUS_IGNORE);
            pos += nameLen;
            std::cout<<"fieldName: "<<fieldNames[f]<<std::endl; //debug

            MPI_File_read_at(fh, pos, &totalMetaBytes[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
            MPI_File_read_at(fh, pos, &totalDataBytes[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
            std::cout<<"totalMetaBytes: "<<totalMetaBytes[f]<<std::endl; //debug
            std::cout<<"totalDataBytes: "<<totalDataBytes[f]<<std::endl; //debug

            uint64_t nRF = 0;
            MPI_File_read_at(fh, pos, &nRF, 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);

            perFieldNumElems[f].resize(nRF);
            perFieldPtsPerElem[f].resize(nRF);
            perFieldCompFlag[f].resize(nRF);

            for (uint64_t r = 0; r < nRF; ++r) {
                MPI_File_read_at(fh, pos, &perFieldNumElems[f][r], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
                MPI_File_read_at(fh, pos, &perFieldPtsPerElem[f][r], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
                MPI_File_read_at(fh, pos, &perFieldCompFlag[f][r], 1, MPI_UNSIGNED_CHAR, MPI_STATUS_IGNORE); pos += sizeof(uint8_t);
            }
            std::cout<<"nRF: "<<nRF<<std::endl;
            std::cout<<"perFieldNumElems: ";
            for (uint64_t r = 0; r < nRF; ++r) {std::cout<<perFieldNumElems[f][r]<<" ";}
            std::cout<<std::endl;
            std::cout<<"perFieldPtsPerElem: ";
            for (uint64_t r = 0; r < nRF; ++r) {std::cout<<perFieldPtsPerElem[f][r]<<" ";}
            std::cout<<std::endl;
            std::cout<<"perFieldCompFlag: ";
            for (uint64_t r = 0; r < nRF; ++r) {std::cout<<static_cast<int>(perFieldCompFlag[f][r])<<" ";}
            std::cout<<std::endl;


            MPI_File_read_at(fh, pos, &fieldMetaOffset[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
            MPI_File_read_at(fh, pos, &fieldDataOffset[f], 1, MPI_UINT64_T, MPI_STATUS_IGNORE); pos += sizeof(uint64_t);
            std::cout<<"fieldMetaOffset: "<<fieldMetaOffset[f]<<std::endl; //debug
            std::cout<<"fieldDataOffset: "<<fieldDataOffset[f]<<std::endl; //debug
        }
    }

    // broadcast to all ranks
    for (int f=0; f<numFields; ++f) {
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

    std::vector<CompressedFieldPerRank> out(numFields);

    for (int f = 0; f < numFields; ++f) {
        auto &CF = out[f];

        // set field name
        CF.name = fieldNames[f];

        // set number of blocks in local rank
        uint64_t nSimRanks = (uint64_t)perFieldNumElems[f].size();
        size_t myBlocks = 0;
        if (nRanks >= nSimRanks) {
            myBlocks = (rank < nSimRanks ? 1 : 0);
        } else {
            myBlocks = nSimRanks / nRanks;
            if(rank == 0) myBlocks = nSimRanks - myBlocks * (nRanks - 1);
        }
        CF.numBlocks = myBlocks;

        // set per-block compressed sizes
        uint64_t myMetaBytes = myBlocks * sizeof(uint64_t);
        uint64_t myMetaDispl = 0;
        MPI_Exscan(&myMetaBytes, &myMetaDispl, 1, MPI_UINT64_T, MPI_SUM, comm);
        if (rank == 0) myMetaDispl = 0;
        CF.compressedSizes.resize(myBlocks);
        if (myBlocks) {
            MPI_File_read_at_all(fh, fieldMetaOffset[f] + myMetaDispl,
                                 CF.compressedSizes.data(), myBlocks, MPI_UINT64_T,
                                 MPI_STATUS_IGNORE);
        }
        
        // set number of points per block & compression flags for each block
        for (int i = myMetaDispl / sizeof(uint64_t); i < myMetaDispl / sizeof(uint64_t) + myBlocks; ++i) {
            CF.ptsPerBlock.push_back(perFieldPtsPerElem[f][i]);
            CF.compressionFlags.push_back(perFieldCompFlag[f][i]);
        } 

        // set per-block offsets inside bytes
        CF.blockOffsets.resize(myBlocks);
        uint64_t offset = 0;
        for (size_t b = 0; b < myBlocks; ++b) {
            CF.blockOffsets[b] = offset;
            offset += CF.compressedSizes[b];
        }

        // set total compressed size of the local rank
        CF.compressedSize = offset;
    }

    // --- Read per-field payload ---
    for (int f=0; f<numFields; ++f) {
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

    // sanity check
    std::vector<uint64_t> allRanksDataBytes(numFields);
    for (int f=0; f<numFields; ++f) {
        uint64_t myData = out[f].bytes.size();
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
    return out;
}


template <typename T = double>
std::vector<T> decompressFieldMPI(const CompressedFieldPerRank &CF, MPI_Comm comm) {
    int rank, nRanks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nRanks);

    uint64_t totalPts = 0;
    for (int i = 0; i < CF.numBlocks; i++) {
        totalPts += CF.ptsPerBlock[i];
    }
    std::vector<T> decompressed(totalPts);

    size_t index = 0;
    for (size_t b = 0; b < CF.numBlocks; ++b) {
        if(CF.compressionFlags[b] == 0) {
            const unsigned char* compPtr = CF.bytes.data() + CF.blockOffsets[b];
            uint64_t compSize = CF.compressedSizes[b];
            uint64_t numT = CF.ptsPerBlock[b];
            assert(compSize % sizeof(T) == 0 && compSize / sizeof(T) == numT);
            for (size_t i = 0; i < numT; ++i) {
                // reinterpret bytes as T
                decompressed[index + i] = reinterpret_cast<const T*>(compPtr)[i];
            }
            index += numT;
        } else {
            try {
                const unsigned char* compPtr = CF.bytes.data() + CF.blockOffsets[b];
                uint64_t compSize = CF.compressedSizes[b];
                uint64_t numT = CF.ptsPerBlock[b];
                T* decompressedPtr = (T*) SZ_decompress(
                    std::is_same<T,double>::value ? SZ_DOUBLE : SZ_FLOAT,
                    const_cast<unsigned char*>(compPtr),
                    compSize,
                    0, 0, 0, 1, numT);
                for (size_t i = 0; i < numT; ++i) {
                    // reinterpret bytes as T
                    decompressed[index + i] = decompressedPtr[i];
                }
                free(decompressedPtr);
                index += numT;
            } catch (const std::invalid_argument& e) {
                std::cerr << "Caught exception: " << e.what() << std::endl;
            }
        }
    }

    assert(index == totalPts);
    return decompressed;
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

    std::vector<CompressedFieldPerRank> all_fields = readCompressedFieldMPI(filename, MPI_COMM_WORLD);


    if (world_rank == 0) {
        std::cout << "File read complete."<<std::endl;
    }

    auto decomp0 = decompressFieldMPI<double>(all_fields[0], MPI_COMM_WORLD);
    std::vector<unsigned char>().swap(all_fields[0].bytes); // free the data of field
    auto decomp1 = decompressFieldMPI<double>(all_fields[1], MPI_COMM_WORLD);
    std::vector<unsigned char>().swap(all_fields[1].bytes);
    auto decomp2 = decompressFieldMPI<double>(all_fields[2], MPI_COMM_WORLD);
    std::vector<unsigned char>().swap(all_fields[2].bytes);
    auto decomp3 = decompressFieldMPI<double>(all_fields[3], MPI_COMM_WORLD);
    std::vector<unsigned char>().swap(all_fields[3].bytes);
    auto decomp4 = decompressFieldMPI<double>(all_fields[4], MPI_COMM_WORLD);
    std::vector<unsigned char>().swap(all_fields[4].bytes);
    auto decomp5 = decompressFieldMPI<double>(all_fields[5], MPI_COMM_WORLD);
    std::vector<unsigned char>().swap(all_fields[5].bytes);

    MPI_Finalize();
    return 0;
}