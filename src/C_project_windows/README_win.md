https://pytorch.org/tutorials/advanced/cpp_export.html
AND
https://pytorch.org/cppdocs/installing.html#minimal-example


Cmake compile On windows:

First:
Find cmake tools;
mkdir build;
cd build;

Then:
..\support_tools\cmake-3.16.2-win64-x86\bin\cmake.exe -DCMAKE_PREFIX_PATH=D:\Ubuntu\libtorch_cpu_win\ ..

Last:
cmake --build .   --config Release       #!!!!!!!!Release
Pit Reminder:
--------------------1 Constexpr requires C++11 feature, use VC-CTP(customer technology preview), Windows only have C++11 version of libtorch.
you'll need to change your "Platform Toolset" to "Visual C++ Compiler Nov 2013 CTP (CTP_Nov2013)" to use the new compiler. You can do that by opening your project's "Property Pages" And going to: "Configuration Properties" > "General" and then changing the "Platform Toolset".
--------------------2 Yet it can recongnize constexpr, it still cannot recongnize std::removexxx within constexpr function. So, I just upgrade VS to 2019!! And it was solved!
--------------------3 Last BUT NOT LEAST, On Windows, debug and release builds are not ABI-compatible. If you plan to build your project in debug mode, please try the debug version of LibTorch. Also, make sure you specify the correct configuration in the cmake --build . line below.

~~~~~~~~~~~~~~~~~~~~~upon, It should compiled well.




Migration:
1 Copy all release file.
2 Copy other dependencies, they may from system (and may from Visual studio). I employeed "Dependencies Walker" to find all dlls. NOTE: some of them may not be shown when first time anaylize .exe, actually, it is after I placed some dll I mannually found from system (mind x86 or x64), it shows some further dependence.