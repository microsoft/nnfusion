mainTest = executable('mainTest', 'test.cpp', include_directories : inc,
        dependencies : doctest)
test('mainTest', mainTest)

colorTest = executable('colorTest', 'colorTest.cpp', include_directories : inc)
test('colorTest', colorTest)

envTermMissing = executable('envTermMissing', 'envTermMissing.cpp', include_directories : inc)
test('envTermMissing', envTermMissing)
