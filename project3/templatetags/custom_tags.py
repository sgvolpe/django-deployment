from django import template
register = template.Library()

from ..models import YourModel

@register.simple_tag
def any_function():
	return YourModel.objects.count()


@register.filter
def split(value):
	return 'asd'
