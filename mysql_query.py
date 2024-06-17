import mysql_setup

cursor = mysql_setup.cursor


def insert_building(building, side, width, risk, crack_id, date):
    query = "INSERT INTO Building (Building, Side, Width, Risk, Crack_id, Date) VALUES (%s,%s,%s,%s,%s,%s)"
    values = (building, side, width, risk, crack_id, date)
    cursor.execute(query, values)


def insert_location(crack_id, latitude, longitude, altitude):
    query = "INSERT INTO Location (Crack_id,Latitude,Longitude,Altitude) VALUSE(%s,%s,%s,%s)"
    values = (crack_id, latitude, longitude, altitude)
    cursor.execute(query, values)


def insert_crackURL(crack_id, crop_url, bbox_url):
    query = "INSERT INTO Crack_url (Crack_id, Crop_url, Bbox_url) VALUES(%s,%s,%s)"
    values = (crack_id, crop_url, bbox_url)
    cursor.execute(query, values)


