from django import template
import datetime
register = template.Library()



@register.filter
def split(value, char):
	return value.split(char)

@register.filter
def get_range(value):
	return list(range(len(value)))

@register.filter
def get_element(value,idx):
	return value[idx]


@register.filter
def parse_date(value):
    return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S").strftime("%d%b %H:%M")
