from flask import Flask, render_template, request, redirect, session,url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, InputRequired, NumberRange
import sklearn
import joblib
import numpy as np

def create_app():
    app = Flask(__name__)

    bootstrap = Bootstrap(app)
    app.config["SECRET_KEY"] = "hard to guess string"

    class EnterYourInfos(FlaskForm):
        age = IntegerField("Enter your Age", validators=[
                           DataRequired(), NumberRange(min=0, max=100)])
        total_Hours = IntegerField("how many hours have you played ?", validators=[
                                   DataRequired(), NumberRange(min=1, max=10000000000)])
        apm = FloatField("what is your APM ?")

        HoursPerWeek = IntegerField("Your  number of hours per weeks")

        SelectByHotkeys = FloatField("Number of unit or building selections made using hotkeys per timestamp")

        UniqueHotkeys = IntegerField("Number of unique hotkeys used per timestamp ")
        
        NumberOfPACs = FloatField("Number of PACs per timestamp")

        GapBetweenPACs = FloatField("Mean duration in milliseconds between PACs")

        ActionLatency = FloatField("Mean latency from the onset of a PACs to their first action in milliseconds")

        TotalMapExplored = FloatField("The number of 24x24 game coordinate grids viewed by the player per timestamp")

        submit = SubmitField("Submit")

    @app.route('/')
    def homepage():
        return render_template('homepage.html')

    @app.route('/about')
    def about():
        return 'This is the about page'

    @app.route('/hello/')
    @app.route('/hello/<name>')
    def hello(name='diallo'):
        return render_template('hello.html', name=name)

    @app.route('/predict', methods=["GET", "POST"])
    def prediction():
        form = EnterYourInfos()
        if request.method == "POST" and form.validate_on_submit():
            # do some stuff
            import sklearn
            import joblib
            from sklearn.ensemble import RandomForestRegressor
            
            model = joblib.load('model_joblib')
            features = [[form.age.data, form.HoursPerWeek.data, form.total_Hours.data, form.apm.data
                                , form.SelectByHotkeys.data, form.UniqueHotkeys.data, form.NumberOfPACs.data, form.GapBetweenPACs.data, form.ActionLatency.data,
                                  form.TotalMapExplored.data]]

            model.predict(np.array(features))
            session["result"] = model.predict(np.array(features).tolist()).tolist()

            return redirect("results")  
        return render_template("prediction_form.html", form=form)

    @app.route("/results")
    def show_result():
        prediction = session["result"]
        if (prediction[0]+0.5) >= (int(prediction[0])+1):
            prediction[0]=int(prediction[0])+1
        else:
            prediction[0]=int(prediction[0])


        return render_template("results.html", prediction = prediction)

    return app
