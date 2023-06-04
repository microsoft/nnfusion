Please run the following commands: (NOTE: please do not run them simultaneously)
```bash
# in tf-ae docker 
cd Figure20 && ./run_tf.sh # about 10 min
logout
# in jax-ae docker
cd Figure20 && ./run_jax.sh # about 10 min
logout
# in cocktailer-ae docker
cd Figure20 && ./run_in_sys_docker.sh # about 1 hour
```