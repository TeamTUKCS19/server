import pymysql


Host = "crack-mysql.cjiec444ylcd.ap-northeast-2.rds.amazonaws.com"
Port = 3306
Username = "admin"
Database = "crack_database"
Password = "teamtukcs19"


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


conn, cursor = connect_RDS(Host, Username, Password, Database, Port)
