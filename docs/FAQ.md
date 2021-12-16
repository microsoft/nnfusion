Frequently problem:
1. Building NNFusion with difference Ubuntu version other than 16.04 or 18.04:
  Currenlty we didn't test Ubuntu 20.04 or orther version. Due to difference release version of Ubuntu comes with different version of build toolchain: This may cause building failure due to compiler's language feature.

2. Building NNFusion with system Anaconda installed:
  This may causes building failure when the libs installed in anaconda don't meet our requirement. E.g: The protobuf installed in Anaconda Env is 3.5.1 but system has a 3.6.0 proto compiler, which will bring failure due to inconsistant of protobuffer's version.