from flask import Flask, render_template, url_for
app=Flask(__name__)


#export FLASK_APP=blog.py
#flask run
#export FLASK_DEBUG=1--debug mode update without exit


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
def home():
    return render_template('home.html',posts=posts, title="Pepper: Blog, Share, More")


@app.route("/explore")
def explore():
    return render_template('explore.html',posts=posts)

# if __name__=="__main__":
#     app.run(debug=True)