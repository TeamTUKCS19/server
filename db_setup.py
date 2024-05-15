from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Building(db.Model):
    __tablename__ = 'building'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(32), nullable=False)
    walls = db.relationship('Wall', backref='building', lazy=True)


class Wall(db.Model):
    __tablename__ = 'wall'
    id = db.Column(db.Integer, primary_key=True)
    building_id = db.Column(db.Integer, db.ForeignKey('building.id', ondelete='CASCADE'), nullable=False)
    direction = db.Column(db.String(32), nullable=False)
    drones = db.relationship('DroneData', backref='wall', lazy=True)


class DroneData(db.Model):
    __tablename__ = 'drone_data'
    id = db.Column(db.Integer, primary_key=True)
    wall_id = db.Column(db.Integer, db.ForeignKey('wall.id', ondelete='CASCADE'), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    altitude = db.Column(db.Float, nullable=False)
    s3_url = db.Column(db.String(255), nullable=False)


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
