Please run the following commands: (NOTE: please do not run them simultaneously)
```bash
# in tf-ae docker 
ssh root@impreza0 -p 31702
cd Figure19 && ./run_tf.sh # about 10 min
logout
# in jax-ae docker
ssh root@impreza0 -p 31703
cd Figure19 && ./run_jax.sh # about 10 min
logout
# in cocktailer-ae docker
ssh root@impreza0 -p 31705
cd Figure19 && ./run_in_sys_docker.sh # about 1 hour
```