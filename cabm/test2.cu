#include <cassert>
#include <iostream>
#include <vector>

#include "data_struct.cuh"

void test2a()
{
    int numRows = 10;
    int numFields = 3;
    std::vector<int> numValsPerFieldMin = { 1, 2, 3 };
    std::vector<int> numValsPerFieldMax = { 10, 20, 30 };
    
    const auto data3D = genRandData3D(numRows, numFields, numValsPerFieldMin, numValsPerFieldMax);

    std::vector<AbmDataGpu> reqAbmDataGpuList;
    std::vector<AbmDataGpu> docAbmDataGpuList;
    for (int fieldIdx = 0; fieldIdx < numFields; fieldIdx++)
    {
        reqAbmDataGpuList.push_back(AbmDataGpu());
        docAbmDataGpuList.push_back(AbmDataGpu());
        reqAbmDataGpuList.at(fieldIdx).init(data3D, fieldIdx, true);
        docAbmDataGpuList.at(fieldIdx).init(data3D, fieldIdx, true);
    }

    for (uint32_t row = 0; row < data3D.size(); row++)
    {
        for (uint32_t field = 0; field < data3D.at(0).size(); field++)
        {
            if (reqAbmDataGpuList.at(field).getNumVals(row) != data3D.at(row).at(field).size())
            {
                std::cout << "Error at (" << row << ", " << field << "): " << reqAbmDataGpuList.at(field).getNumVals(row) << " != " << data3D.at(row).at(field).size() << std::endl;
                assert(false);
            }
            for (uint32_t valOffset = 0; valOffset < data3D.at(row).at(field).size(); valOffset++)
            {
                ABM_DATA_TYPE val = reqAbmDataGpuList.at(field).getVal(row, valOffset);
                if (val != data3D.at(row).at(field).at(valOffset))
                {
                    std::cout << "Error at (" << row << ", " << field << ", " << valOffset << "): " << val << " != " << data3D.at(row).at(field).at(valOffset) << std::endl;
                    assert(false);
                }
            }
        }
    }

    for (int fieldIdx = 0; fieldIdx < numFields; fieldIdx++)
    {
        reqAbmDataGpuList.at(fieldIdx).free();
        docAbmDataGpuList.at(fieldIdx).free();
    }
}

int main()
{
    test2a();
    return 0;
}