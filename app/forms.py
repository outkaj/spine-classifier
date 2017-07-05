# ourapp/forms.py

from flask_wtf import Form
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Email

class FilepathForm(Form):
    filepath = StringField('Filepath', validators=[DataRequired(), Filepath()])
