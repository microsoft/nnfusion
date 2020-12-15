## Coding Style
The coding style NNFUSION uses is LLVM style. It's recorded in $(nnfusion_root)/.clang-format
### How to Apply coding style for whole project
Install clang-format with 3.9: `sudo apt-get install clang-format-3.9`.   
Use  `cd maint && bash ./apply-code-format.sh` or `make style` to apply coding style.

### How to write comment  
The documents will be mainly based on Doxygen Style comment in source code, which is documented [here](http://www.doxygen.nl/manual/index.html).

## Debug mode
Enable debugging by adding  "-DNNFUSION_DEBUG_ENABLE=TRUE  -DCMAKE_BUILD_TYPE=Debug" to CMake command line;  
 When meet crash, use command "ulimit -c unlimited" to enable core dump;  
Use GDB to debug the program.

## FAQ
### Git EOL Problem
This happens if using Windows as coding enviroment.
Solve this by "checkout as system's style(CRLF or LF), check in as Unix style(LF)":  
`$ git config --global core.autocrlf input`  
`$ git config --global core.safecrlf true`  


### Git Permission Problem caused by Samba
If you use samba to share your disk with your windows pc, then the default configuration will lead to file permission problem:   
Add "create mask = 0664"  and "directory mask = 0775" into the right position in your "/etc/samba/smb.conf" to avoid this problem.  
Because Git thinks the permission is also part of the file.  
Another Solution:  git config core.fileMode false