from nnfusion import Config


def test_config():
    # default
    config = Config()
    assert config['kernel_tuning_steps'] == 1000
    assert config['function_codegen'] == True

    # init with kwargs
    config = Config(kernel_tuning_steps=42,
                    function_codegen=False,
                    foo=True,)
    assert config['kernel_tuning_steps'] == 42
    assert config['function_codegen'] == False
    assert config['foo'] == True

    # init with dict
    config = Config({
        'kernel_tuning_steps': 42,
        'function_codegen': False,
        'foo': True,
    })
    assert config['kernel_tuning_steps'] == 42
    assert config['function_codegen'] == False
    assert config['foo'] == True

