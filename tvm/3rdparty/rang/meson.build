project('rang', 'cpp', version : '3.1.0',
		default_options : ['cpp_std=c++11'])

inc = include_directories('include')

doctest = dependency('doctest')
subdir('test')

run_target('cppcheck', command : ['cppcheck', '--project=' +
  			join_paths(meson.build_root(), 'compile_commands.json')])
