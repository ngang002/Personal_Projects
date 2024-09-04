from datetime import datetime, timedelta

_EPOCHS = 1000
_PATIENCE_EPOCHS = 10

_BATCH_SIZE = 80
_NLAYERS = 4
_ALPHA = 2E-4

_VAL_DAYS = 365


_previous_timesteps = 20
_next_timesteps = 6


def _get_EndDate(offset_today_by_days=0):
	end_date = (datetime.today() - timedelta(days=offset_today_by_days))
	end_date_str = end_date.strftime("%Y-%m-%d")
	return end_date, end_date_str

def _get_date(end_date, days_diff):
	date = end_date-timedelta(days=days_diff)
	date_str = date.strftime("%Y-%m-%d")
	return date, date_str