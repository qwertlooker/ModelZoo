# 此文件用于启动webui.py并增加性能优化环境变量

# 算子下发队列优化
export TASK_QUEUE_ENABLE=2
# 设置是否开启均匀绑核,0-关闭/1-开启粗粒度绑核/2-开启细粒度绑核
export CPU_AFFINITY_CONF=1
# 指定NPU device id
export ASCEND_RT_VISIBLE_DEVICES=0
# 如有libjemalloc.so
export LD_PRELOAD="/disk1/libjemalloc.so $LD_PRELOAD"

unset ASCEND_LAUNCH_BLOCKING

python webui.py