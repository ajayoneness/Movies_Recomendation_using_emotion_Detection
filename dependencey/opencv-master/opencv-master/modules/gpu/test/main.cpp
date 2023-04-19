/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cvtest;
using namespace testing;

void printOsInfo()
{
#if defined _WIN32
#   if defined _WIN64
        cout << "OS: Windows x64 \n" << endl;
#   else
        cout << "OS: Windows x32 \n" << endl;
#   endif
#elif defined linux
#   if defined _LP64
        cout << "OS: Linux x64 \n" << endl;
#   else
        cout << "OS: Linux x32 \n" << endl;
#   endif
#elif defined __APPLE__
#   if defined _LP64
        cout << "OS: Apple x64 \n" << endl;
#   else
        cout << "OS: Apple x32 \n" << endl;
#   endif
#endif
}

void printCudaInfo()
{
#if !defined HAVE_CUDA || defined(CUDA_DISABLER)
    cout << "OpenCV was built without CUDA support \n" << endl;
#else
    int driver;
    cudaDriverGetVersion(&driver);

    cout << "CUDA Driver  version: " << driver << '\n';
    cout << "CUDA Runtime version: " << CUDART_VERSION << '\n';

    cout << endl;

    cout << "GPU module was compiled for the following GPU archs:" << endl;
    cout << "    BIN: " << CUDA_ARCH_BIN << '\n';
    cout << "    PTX: " << CUDA_ARCH_PTX << '\n';

    cout << endl;

    int deviceCount = getCudaEnabledDeviceCount();
    cout << "CUDA device count: " << deviceCount << '\n';

    cout << endl;

    for (int i = 0; i < deviceCount; ++i)
    {
        DeviceInfo info(i);

        cout << "Device [" << i << "] \n";
        cout << "\t Name: " << info.name() << '\n';
        cout << "\t Compute capability: " << info.majorVersion() << '.' << info.minorVersion()<< '\n';
        cout << "\t Multi Processor Count: " << info.multiProcessorCount() << '\n';
        cout << "\t Total memory: " << static_cast<int>(static_cast<int>(info.totalMemory() / 1024.0) / 1024.0) << " Mb \n";
        cout << "\t Free  memory: " << static_cast<int>(static_cast<int>(info.freeMemory() / 1024.0) / 1024.0) << " Mb \n";
        if (!info.isCompatible())
            cout << "\t !!! This device is NOT compatible with current GPU module build \n";

        cout << endl;
    }
#endif
}

int main(int argc, char** argv)
{
    try
    {
         const char*  keys =
                "{ h | help ?            | false | Print help}"
                "{ i | info              | false | Print information about system and exit }"
                "{ d | device            | -1   | Device on which tests will be executed (-1 means all devices) }"
                "{ nvtest_output_level | nvtest_output_level | none | NVidia test verbosity level (none, compact, full) }"
                ;

        CommandLineParser cmd(argc, (const char**)argv, keys);

        if (cmd.get<bool>("help"))
        {
            cmd.printParams();
            return 0;
        }

        printOsInfo();
        printCudaInfo();

        if (cmd.get<bool>("info"))
        {
            return 0;
        }

        int device = cmd.get<int>("device");
        if (device < 0)
        {
            DeviceManager::instance().loadAll();

            cout << "Run tests on all supported devices \n" << endl;
        }
        else
        {
            DeviceManager::instance().load(device);

            DeviceInfo info(device);
            cout << "Run tests on device " << device << " [" << info.name() << "] \n" << endl;
        }

        string outputLevel = cmd.get<string>("nvtest_output_level");

        if (outputLevel == "none")
            nvidiaTestOutputLevel = OutputLevelNone;
        else if (outputLevel == "compact")
            nvidiaTestOutputLevel = OutputLevelCompact;
        else if (outputLevel == "full")
            nvidiaTestOutputLevel = OutputLevelFull;

        TS::ptr()->init("gpu");
        InitGoogleTest(&argc, argv);

        return RUN_ALL_TESTS();
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
        return -1;
    }
    catch (...)
    {
        cerr << "Unknown error" << endl;
        return -1;
    }

    return 0;
}

#else // HAVE_CUDA

int main()
{
    printf("OpenCV was built without CUDA support\n");
    return 0;
}

#endif // HAVE_CUDA
