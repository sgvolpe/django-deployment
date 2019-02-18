# Generated by Django 2.1.1 on 2018-12-10 12:20

from django.conf import settings
import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('new_dash', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='several_same_BFM',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(blank=True, max_length=50)),
                ('description', models.CharField(blank=True, max_length=255)),
                ('ap', models.CharField(blank=True, max_length=500, validators=[django.core.validators.RegexValidator('^([0-9]+,{0,1})+$', 'Check!')])),
                ('los', models.CharField(blank=True, max_length=500, validators=[django.core.validators.RegexValidator('^([0-9]+,{0,1})+$', 'Check!')])),
                ('onds', models.CharField(blank=True, max_length=500, validators=[django.core.validators.RegexValidator('^([A-Z]{3}-[A-Z]{3},)*([A-Z]{3}-[A-Z]{3})+$', 'Check!')])),
                ('bfm_template_1', models.TextField(blank=True, null=True)),
                ('repeats', models.IntegerField()),
                ('total_queries', models.IntegerField(default=0)),
                ('analysis_finished', models.DateTimeField(blank=True, null=True)),
                ('author', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
