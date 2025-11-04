"""Celery application for MotionMatch background tasks"""
from celery import Celery
from motionmatch.core.config import config

# Create Celery app
celery_app = Celery(
    'motionmatch',
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
    include=['motionmatch.workers.tasks']
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    worker_prefetch_multiplier=1,  # Important for GPU tasks
    worker_max_tasks_per_child=100,  # Restart workers to prevent memory leaks
    task_routes={
        'motionmatch.workers.tasks.index_video_task': {'queue': 'indexing'},
        'motionmatch.workers.tasks.batch_index_task': {'queue': 'indexing'},
    }
)

if __name__ == '__main__':
    celery_app.start()