import mysql_setup

# cursor = mysql_setup.cursor


def insert_building(cursor, building, side, width, risk, crack_id, date):
    query = "INSERT INTO Building (Building, Side, Width, Risk, Crack_id, Date) VALUES (%s,%s,%s,%s,%s,%s)"
    values = (building, side, width, risk, crack_id, date)
    try:
        cursor.execute(query, values)
        print("Building inserted successfully!")
    except Exception as e:
        print(f"Error inserting building: {e}")

def insert_location(cursor, crack_id, latitude, longitude, altitude):
    query = "INSERT INTO Location (Crack_id,Latitude,Longitude,Altitude) VALUES(%s,%s,%s,%s)"
    values = (crack_id, latitude, longitude, altitude)
    try:
        cursor.execute(query, values)
        print("location inserted successfully!")
    except Exception as e:
        print(f"Error inserting location: {e}")


def insert_crackURL(cursor, crack_id, crop_url, bbox_url):
    query = "INSERT INTO Crack_url (Crack_id, Crop_url, Bbox_url) VALUES(%s,%s,%s)"
    values = (crack_id, crop_url, bbox_url)
    try:
        cursor.execute(query, values)
        print("Crack_url inserted successfully!")
    except Exception as e:
        print(f"Error inserting Crack_url: {e}")


