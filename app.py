from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/simulation')
def about():
    return render_template('simulation.html')



@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


if __name__ == '__main__':

    # Run this when running on LOCAL server...
    app.run(debug=True)

    # ...OR run this when PRODUCTION server.
    # app.run(debug=False)