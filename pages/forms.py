from django import forms

#id="planetSearch" placeholder="Enter planet or star name..." class="search-input"

class ObjectForm(forms.Form):
    object_id = forms.CharField(max_length=256, label=False, required=False, widget=forms.TextInput(attrs={"id": "planetSearch", "placeholder": "Enter planet or star name...", "class": "search-input", "style": "width:600px;"}))