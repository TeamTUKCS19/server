import pymysql


def connect_RDS(host, username, password, database, port):
    conn = None
    cursor = None
    try:
        conn = pymysql.connect(host=host, user=username, passwd=password, db=database, port=port, use_unicode=True,
                               charset='utf8')
        cursor = conn.cursor()
        print("Connecting to RDS is Success")
    except pymysql.Error as e:
        print(f"Error connecting to MySQL: {e}")

    return conn, cursor


def insert_building(conn, cursor, building, side, width, risk, crack_id, date):
    query = "INSERT INTO Building (Building, Side, Width, Risk, Crack_id, Date) VALUES (%s,%s,%s,%s,%s,%s)"
    values = (building, side, width, risk, crack_id, date)
    try:
        cursor.execute(query, values)
        conn.commit()
        print("Building inserted successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting building: {e}")


def insert_location(conn, cursor, crack_id, latitude, longitude, altitude):
    query = "INSERT INTO Location (Crack_id,Latitude,Longitude,Altitude) VALUES(%s,%s,%s,%s)"
    values = (crack_id, latitude, longitude, altitude)
    try:
        cursor.execute(query, values)
        conn.commit()
        print("location inserted successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting location: {e}")


def insert_crackURL(conn, cursor, crack_id, crop_url, bbox_url):
    query = "INSERT INTO Crack_url (Crack_id, Crop_url, Bbox_url) VALUES(%s,%s,%s)"
    values = (crack_id, crop_url, bbox_url)
    try:
        cursor.execute(query, values)
        conn.commit()
        print("Crack_url inserted successfully!")
    except Exception as e:
        conn.rollback()
        print(f"Error inserting Crack_url: {e}")
