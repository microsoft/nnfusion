#ifndef LOGGER_H
#define LOGGER_H

#include "logging.h"

extern Logger gLogger;
extern LogStreamConsumer gLogVerbose;
extern LogStreamConsumer gLogInfo;
extern LogStreamConsumer gLogWarning;
extern LogStreamConsumer gLogError;
extern LogStreamConsumer gLogFatal;

void setReportableSeverity(Logger::Severity severity);

#endif // LOGGER_H

