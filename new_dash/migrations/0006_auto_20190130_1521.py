# Generated by Django 2.1.1 on 2019-01-30 15:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('new_dash', '0005_virtualinterlining'),
    ]

    operations = [
        migrations.AlterField(
            model_name='virtualinterlining',
            name='ddate',
            field=models.CharField(default='2019-03-31', max_length=20),
        ),
        migrations.AlterField(
            model_name='virtualinterlining',
            name='des',
            field=models.CharField(default='LON', max_length=3),
        ),
        migrations.AlterField(
            model_name='virtualinterlining',
            name='ori',
            field=models.CharField(default='MVD', max_length=3),
        ),
        migrations.AlterField(
            model_name='virtualinterlining',
            name='rdate',
            field=models.CharField(default='2019-04-15', max_length=20),
        ),
    ]