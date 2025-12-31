from celery import Celery

celery_app = Celery(
    'onboarding_api',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Explicit import to register the task
import tasks.inference_worker

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
