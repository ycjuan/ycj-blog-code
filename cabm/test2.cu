#include <cassert>
#include <iostream>
#include <vector>

#include "data_struct.cuh"

void test2a()
{
    std::vector<std::vector<std::vector<ABM_DATA_TYPE>>> data3D = genRandData3D(10, 3, { 1, 2, 3 }, { 10, 20, 30 });

    AbmDataGpu abmDataGpu;
    abmDataGpu.init(data3D, true);

    for (uint32_t row = 0; row < data3D.size(); row++)
    {
        for (uint32_t field = 0; field < data3D.at(0).size(); field++)
        {
            for (uint32_t valOffset = 0; valOffset < data3D.at(row).at(field).size(); valOffset++)
            {
                int fieldOffset = abmDataGpu.getOffset(row, field);
                ABM_DATA_TYPE val = abmDataGpu.getVal(row, fieldOffset + valOffset);
                if (val != data3D.at(row).at(field).at(valOffset))
                {
                    std::cout << "Error at (" << row << ", " << field << ", " << valOffset << "): " << val << " != " << data3D.at(row).at(field).at(valOffset) << std::endl;
                    assert(false);
                }
            }
        }
    }

    abmDataGpu.free();
}

int main()
{
    test2a();
    return 0;
}