SECRET_KEY = "djiaalamcl-dkspmdamac488dda"
CSRF_ENABLED = True
CELERY_BROKER_URL='redis://127.0.0.1:6379/0'
CELERY_RESULT_BACKEND='redis://127.0.0.1:6379/0'
CELERY_TIMEZONE = 'Asia/Shanghai'
CELERY_ENABLE_UTC = False
CELERYD_FORCE = True
CELERYD_CONCURRENCY = 4 #worker个数
CELERYD_PREFETCH_MULTIPLIER = 4 #同时预取得消息个数
CELERYD_FORCE_EXECV = True
CELERYD_MAX_TASKS_PER_CHILD = 100
CELERYD_TASK_SOFT_TIME_LIMIT = 10