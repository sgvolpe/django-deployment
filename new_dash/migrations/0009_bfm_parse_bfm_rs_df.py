# Generated by Django 2.1.1 on 2019-02-13 16:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('new_dash', '0008_bfm_parse'),
    ]

    operations = [
        migrations.AddField(
            model_name='bfm_parse',
            name='bfm_rs_df',
            field=models.CharField(blank=True, max_length=255),
        ),
    ]