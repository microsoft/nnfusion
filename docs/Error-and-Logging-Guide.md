## Error Handling
Some common errors have been wrapped in "nnfusion/util/errors.hpp", you can use them by throw these wrapping errors, please do not throw unwrapped errors directly in our code, if there are some errors unsupported, you can add them in "nnfusion/util/errors.hpp";
	
Usage:
`throw nnfusion::errors::RuntimeError("message");`

we provide the following check interfaces, if check failed, an error log will be recored and sepecified exception (default is nnfusion::errors::CheckError) will be thrown
	
`CHECK(condition) << "message"; // Check condition, throw CheckError exception if fails`  
`CHECK_FAIL() << "message"; // log error message and throw CheckError exception`  
`CHECK_WITH_EXCEPTION(condition, T) << "message"; //Check condition with an exception class of "T"`  
`CHECK_FAIL_WITH_EXCEPTION(T) << "message";   // log error message, and throw exception "T"`  
`CHECK_NOT_NULLPTR(ptr_) << "message"; // Check ptr_ isn't nullptr, throw   nnfusion::errors::NullPointer, if fails`  
	
Also, DCHECK is provided, these checks are ignored if debug isn't enable (the same as "assert")
	
`DCHECK(condition) << "message"; // Check condition, throw CheckError exception if fails`  
`DCHECK_FAIL() << "message"; // log error message and throw CheckError exception`  
`DCHECK_WITH_EXCEPTION(condition, T) << "message"; //Check condition with an exception class of "T"`  
`DCHECK_FAIL_WITH_EXCEPTION(T) << "message"; // log error message, and throw exception "T"`  
`DCHECK_NOT_NULLPTR(ptr_) << "message"; // Check ptr_ isn't nullptr, throw nnfusion::errors::NullPointer, if fails`  


## Logging 
Include the util file when you want to use error handling and logging:  
`#include "nnfusion/util/util.hpp"`   

There are 5 level for logging, you can control print info by setting env "MIN_LOG_LEVEL" (default 1, INFO) 
	
	    const int DEBUG = 0;
	    const int INFO = 1;
	    const int WARNING = 2;
	    const int ERROR = 3;
	    const int FATAL = 4; # FATAL is a placeholder, and don't used now.
	
Usage (WARNING as a example):   
`LOG(WARNING) << "message";`
		
Control print logging level:   
	`MIN_LOG_LEVEL =2  ./src/tools/nnfusion/nnfusion frozen_model.pb -f tensorflow`

