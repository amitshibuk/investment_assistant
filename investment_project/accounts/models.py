from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):

    ROLE_CHOICES = (
        ('admin', 'Admin'),
        ('premium', 'Premium'),
        ('normal', 'Normal'),
    )

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='normal')
    phone = models.CharField(max_length=15, blank=True, null=True)
    subscription_active = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.user.username} Profile"
