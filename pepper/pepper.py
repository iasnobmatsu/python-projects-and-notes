from flask import Flask, render_template, url_for, flash, redirect
from forms import RegistrationForm, LoginForm
app=Flask(__name__)

app.config['SECRET_KEY']="9a62337156cabd15a1744c7a1706c4a1"

#export FLASK_APP=blog.py
#flask run
#export FLASK_DEBUG=1--debug mode update without exit



#pip3 install flask_wtf



#dummy data
posts=[
    {
        "user":"Ramon Cajal",
        "title":"Purkinje Cell Structure",
        "content":"Mitch Prinstein is the John Van Seters Distinguished Professor of Psychology and Neuroscience, and a member of the Clinical Psychology Program. Mitch’s research uses a developmental psychopathology framework to understand how adolescents’ interpersonal experiences, particularly among peers,  are associated with depression, self-injury, and health risk behaviors.  Mitch’s work has two areas of focus.",
        "date":"May 31, 1880"
    },{
        "user":"Mitch Prinstein",
        "title":"Peer Relations",
        "content":"rrrrrrrrrr\nr\n\n\n\n\n\rrr",
        "date":"August 2, 2019"
    }
]


@app.route("/")
@app.route("/homw")
def home():
    return render_template('home.html',posts=posts, title="Pepper: Blog, Share, More")


@app.route("/explore")
def explore():
    return render_template('explore.html', title="Pepper:Explore", posts=posts)


@app.route("/signup", methods=['GET','POST'])
def signup():
    form=RegistrationForm()
    if form.validate_on_submit():
        flash("Signed up successfully for {}".format(form.username.data),"success")
        return redirect(url_for('home'))
    return render_template("signup.html", title="Pepper: Sign Up", form=form)


@app.route("/login",methods=['GET','POST'])
def login():
    form=LoginForm()
    if form.validate_on_submit():
        if form.email.data=="admin@pepper.com" and form.password.data=="pepper":
            flash("Login successfully for {}".format(form.email.data),"success")
            return redirect(url_for('home'))
        else:
            flash("Login Unsuccessful for {}".format(form.email.data),"danger")
    return render_template("login.html", title="Pepper: Login", form=form)

# if __name__=="__main__":
#     app.run(debug=True)