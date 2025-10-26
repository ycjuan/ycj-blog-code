#include <cassert>
#include <iostream>
#include <vector>

#include "data_struct.cuh"

void test2a()
{
    std::vector<std::vector<std::vector<long>>> data3D = genRandData3D(10, 3, { 1, 2, 3 }, { 10, 20, 30 });

    AbmDataGpu abmDataGpu;
    abmDataGpu.init(data3D, true);

    for (uint32_t row = 0; row < data3D.size(); row++)
    {
        for (uint32_t field = 0; field < data3D.at(0).size(); field++)
        {
            for (uint32_t valOffset = 0; valOffset < data3D.at(row).at(field).size(); valOffset++)
            {
                int fieldOffset = abmDataGpu.getOffset_d(row, field);
                long val = abmDataGpu.getVal_d(row, fieldOffset + valOffset);
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