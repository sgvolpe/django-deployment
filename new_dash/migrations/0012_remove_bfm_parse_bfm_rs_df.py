# Generated by Django 2.1.1 on 2019-02-13 16:45

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('new_dash', '0011_bfm_parse_bfm_rs_df'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='bfm_parse',
            name='bfm_rs_df',
        ),
    ]
