project('rang', 'cpp', version : '3.1.0',
		default_options : ['cpp_std=c++11'])

inc = include_directories('include')

visualTest = executable('visualTest', 'visualTest.cpp', include_directories : inc)
test('visualTest', visualTest)
