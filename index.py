import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.options
from tornado.options import define, options
import pandas as pd
import models
import os.path

# handler for uploading the dataset
class uploadHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("upload.html")

    async def post(self):
        files = self.request.files["dataFile"]
        for f in files:
            fh = open(f"uploads/{f.filename}", "wb")
            fh.write(f.body)
            fh.close()
        df = pd.read_csv(f'uploads/{f.filename}')
        df = df.dropna().reset_index(drop=True)
        dfHTML = df.to_html(max_rows=15, justify='center', col_space=50)
        self.render("train.html", filename=f.filename, data=dfHTML, trained=False, acc=0)

# handler for training the logistic regression model    
class trainHandler(tornado.web.RequestHandler):
    async def get(self, filename):
        df = pd.read_csv(f'uploads/{filename}')
        col_predict = int(self.get_argument("col"))
        df, x, y = models.getSets(df, col_predict)
        dfHTML = df.to_html(max_rows=15, justify='center', col_space=50)
        reg = models.LogisticRegression()
        acc = await reg.trainAndPredict(x, y, 0.5)
        self.render("train.html", filename=filename, data=dfHTML, trained=True, acc=acc, col=col_predict)

# handler for generating and visualizing the ROC curve
class visualizeHandler(tornado.web.RequestHandler):
    async def get(self, filename, col_predict):
        df = pd.read_csv(f'uploads/{filename}')
        df, x, y = models.getSets(df, int(col_predict))
        curve = models.ROC_Curve()
        tpr, fpr, matrices = await curve.getCurve(x, y)
        self.render("visualize.html", tpr=tpr, fpr=fpr, matrices=matrices)


# create and start application
if __name__ == "__main__":
    app = tornado.web.Application([
        ("/", uploadHandler),
        ("/train/(.*)", trainHandler),
        ("/visualize/(.*)/([0-9]+)", visualizeHandler)
        ],
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        debug=True
    )

    app.listen(8888)
    print("Listening on port 8888")

    tornado.ioloop.IOLoop.instance().start()
