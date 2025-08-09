import logging
import logging.config  # uv pip install logging -i https://pypi.tuna.tsinghua.edu.cn/simple


# 创建一个过滤器
class MyFilter(logging.Filter):
    def filter(self, record):
        # 如果日志信息中包含"Using selector: EpollSelector"，则阻止这条日志的输出
        return "Using selector" not in record.getMessage() and "EpollSelector" not in record.getMessage()

# 读取配置文件
logging.config.fileConfig('src/mylogging.conf')
logger = logging.getLogger('applog')
# 添加自定义的过滤器
logger.addFilter(MyFilter())


# 使用不同级别的日志输出
logger.debug('This is a debug message')
logger.info('This is an info message')
logger.warning('This is a warning message')
logger.error('This is an error message')
logger.critical('This is a critical message')
logger.debug('Using selector: EpollSelector')


if __name__ == '__main__':
    pass