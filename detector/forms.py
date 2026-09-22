from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError

class UserRegistrationForm(forms.ModelForm):
    """
    User registration form with username, email, password, and confirmation.
    """
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your username',
            'required': True,
            'autocomplete': 'username',
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your email address',
            'required': True,
            'autocomplete': 'email',
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Create a strong password (min 6 characters)',
            'required': True,
            'autocomplete': 'new-password',
        })
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Confirm your password',
            'required': True,
            'autocomplete': 'new-password',
        })
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username__iexact=username).exists():
            raise ValidationError("This username is already registered. Please choose another one.")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email__iexact=email).exists():
            raise ValidationError("An account with this email address already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        password = cleaned_data.get('password')
        confirm_password = cleaned_data.get('confirm_password')

        if password and confirm_password:
            if password != confirm_password:
                self.add_error('confirm_password', "Passwords do not match.")
            if len(password) < 6:
                self.add_error('password', "Password must be at least 6 characters long.")
        return cleaned_data


class UserLoginForm(forms.Form):
    """
    Login form allowing username or email login.
    """
    username_or_email = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your username or email',
            'required': True,
            'autocomplete': 'username',
        }),
        label="Username or Email"
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter your password',
            'required': True,
            'autocomplete': 'current-password',
        }),
        label="Password"
    )


class NewsSubmissionForm(forms.Form):
    """
    News headline / article input form with validation and engine toggles.
    """
    news_title = forms.CharField(
        required=False,
        max_length=350,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Optional: Enter news headline or claim title...',
            'id': 'newsInputTitle'
        }),
        label="Headline / Title (Optional)"
    )
    news_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'class': 'form-control news-textarea',
            'rows': 6,
            'placeholder': 'Paste a news headline, claim statement, or full article text here to verify...',
            'id': 'newsInputText',
            'required': True,
        }),
        label="News Headline or Article Content",
        min_length=10,
        max_length=25000,
        error_messages={
            'required': 'Please enter a news headline or article to analyze.',
            'min_length': 'Please enter at least 10 characters for a meaningful prediction.',
            'max_length': 'The text exceeds the maximum permitted length of 25,000 characters.',
        }
    )
    skip_live_verification = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input', 'id': 'skipLiveCheck'}),
        label="Run ML-Only Mode (Skip external live web search)"
    )

    def clean_news_text(self):
        text = self.cleaned_data.get('news_text', '').strip()
        if not text:
            raise ValidationError("News text cannot be empty or purely whitespace.")
        if len(text.split()) < 3:
            raise ValidationError("Please provide at least 3 words for accurate analysis.")
        return text
