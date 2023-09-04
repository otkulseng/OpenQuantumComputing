import structlog
LOG = structlog.get_logger(file=__name__)

# import logging
# import structlog

# structlog.configure(
#     wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
# )

def func1():
    log = LOG.bind(func=func1.__name__)

    log.debug("This is a log msg")
    log.error("Failed to upload")
    log.info("Helo")
    log.warn("h")

def func2():
    log = LOG.bind(func=func2.__name__)

    log.debug("This is another log msg")
    log.error("Failed to upload")



def main():
    func1()
    func2()

