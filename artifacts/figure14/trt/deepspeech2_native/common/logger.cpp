#include "logger.h"
#include "logging.h"

//! Decouple default reportability level for sample and TRT-API specific logging.
//! This change is required to not log TRT-API specific info by default.
//! To enable verbose logging of TRT-API use `setReportableSeverity(Severity::kINFO)`.

//! TODO: Revert gLoggerSample to gLogger to use same reportablilty level for TRT-API and samples
//! once we have support for Logger::Severity::kVERBOSE. TensorRT runtime will enable this
//! new logging level in future releases when making other ABI breaking changes.

//! gLogger is used to set default reportability level for TRT-API specific logging.
Logger gLogger{Logger::Severity::kWARNING};

//! gLoggerSample is used to set default reportability level for sample specific logging.
Logger gLoggerSample{Logger::Severity::kINFO};

LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLoggerSample)};
LogStreamConsumer gLogInfo{LOG_INFO(gLoggerSample)};
LogStreamConsumer gLogWarning{LOG_WARN(gLoggerSample)};
LogStreamConsumer gLogError{LOG_ERROR(gLoggerSample)};
LogStreamConsumer gLogFatal{LOG_FATAL(gLoggerSample)};

void setReportableSeverity(Logger::Severity severity)
{
    gLogger.setReportableSeverity(severity);
    gLogVerbose.setReportableSeverity(severity);
    gLogInfo.setReportableSeverity(severity);
    gLogWarning.setReportableSeverity(severity);
    gLogError.setReportableSeverity(severity);
    gLogFatal.setReportableSeverity(severity);
}
