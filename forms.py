from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import DataRequired

class FilepathForm(FlaskForm):
    filepath = StringField('Filepath', validators=[DataRequired()])
