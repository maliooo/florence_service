cd /home/zhangxuqi/zhangxuqi/florence_service
device_id=0
port=20093
python_path=/home/zhangxuqi/miniconda3/envs/malio/bin/python
nohup $python_path src/florence_api.py --device_id $device_id --port $port 2>&1 |  ts '[%Y-%m-%d %H:%M:%S]' | cronolog logs/${port}_%Y-%m-%d.log &
