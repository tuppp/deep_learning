import numpy as np
import sqlalchemy as sqla

from sqlalchemy import *
from sqlalchemy.ext.declarative import declarative_base


#Path to userdata file
path_to_login_file = "login_data"


def compare_average_temp(Dwd, table, se):
    r"""
    returns table data with columns station name, postcode, measure date, Dwd temp, table temp
    """
    return getResult(se.query(Dwd.c.station_name, Dwd.c.postcode, Dwd.c.measure_date, Dwd.c.average_temp, table.c.temp).filter(Dwd.c.measure_date == table.c.measure_date).filter(Dwd.c.postcode == table.c.postcode), se)

def compare_clouds(Dwd, table, se):
    r"""
    :return: table data with colums station name, postcode, measure date, Dwd, clouds
    """
    return getResult(se.query(Dwd.c.station_name, Dwd.c.postcode, Dwd.c.measure_date, Dwd.c.coverage_amount, table.c.clouds).filter(Dwd.c.measure_date == table.c.measure_date).filter(Dwd.c.postcode == table.c.postcode), se)

def compare_humidity(Dwd, table, se):
    r"""
    :return: table data with colums station name, postcode, measure date, Dwd, clouds
    """
    return getResult(se.query(Dwd.c.station_name, Dwd.c.postcode, Dwd.c.measure_date, Dwd.c.relative_himidity, table.c.humidity_prob).filter(Dwd.c.measure_date == table.c.measure_date).filter(Dwd.c.postcode == table.c.postcode), se)

def compare_average_temp(Dwd, table, se):
    r"""
    returns table data with columns station name, postcode, measure date, Dwd temp, table temp

    :param Dwd: dwd table
    :param table: another table to compare
    :param se: session object containing connection information
    :returns: result table
    """
    return getResult(se.query(Dwd.c.station_name, Dwd.c.postcode, Dwd.c.measure_date, Dwd.c.average_temp, table.c.temp).filter(Dwd.c.measure_date == table.c.measure_date, Dwd.c.postcode == table.c.postcode), se)

def compare_air_pressure(Dwd, table, se):
    r"""
    returns table data with columns station name, postcode, measure date, Dwd air_pressure, table air_pressure_ground

    :param Dwd: dwd table
    :param table: another table to compare
    :param se: session object containing connection information
    :returns: result table
    """
    return getResult(se.query(Dwd.c.station_name, Dwd.c.postcode, Dwd.c.measure_date, Dwd.c.air_pressure, table.c.air_pressure_ground).filter(Dwd.c.measure_date == table.c.measure_date, Dwd.c.postcode == table.c.postcode), se)

def compare_wind_speed(Dwd, table, se):
    r"""
    returns table data with columns station name, postcode, measure date, Dwd average_wind_speed, table wind_speed

    :param Dwd: dwd table
    :param table: another table to compare
    :param se: session object containing connection information
    :returns: result table
    """
    return getResult(se.query(Dwd.c.station_name, Dwd.c.postcode, Dwd.c.measure_date, Dwd.c.average_wind_speed, table.c.wind_speed).filter(Dwd.c.measure_date == table.c.measure_date, Dwd.c.postcode == table.c.postcode), se)

def compare_sun_hours(Dwd, table, se):
    r"""
    returns table data with columns station name, postcode, measure date, Dwd sun_hours, table sun_hours

    :param Dwd: dwd table
    :param table: another table to compare
    :param se: session object containing connection information
    :returns: result table
    """
    return getResult(se.query(Dwd.c.station_name, Dwd.c.postcode, Dwd.c.measure_date, Dwd.c.sun_hours, table.c.sun_hours).filter(Dwd.c.measure_date == table.c.measure_date, Dwd.c.postcode == table.c.postcode), se)

# Functions
def getPostcode(postcode, table, se, query=None):
    r"""
    Method gets all entries with given postcode
    :param postcode: desired postcode
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        query = se.query(table).filter(table.c.postcode == postcode)
        return query
    else:
        query = query.filter(table.c.postcode == postcode)
        return query


def getRain(table, se, query=None):
    r"""
        Returns rain information
       :param table: which weather table is going to be used
       :param se: Session Object containing connection information
       :param query: Query Object which contains SQL query, if empty one will be created
       :returns: Query Object, can be reused for other queries
       """
    if (query == None):
        query = se.query(table).filter(table.c.precipitation_type != "kein NS")
        return query
    else:
        query = query.filter(table.c.precipitation_type != "kein NS")
        return query


def getmaxSunHours(table, se, query=None):
    r"""
    Returns table Data with the most sun hours

    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        return se.query(table, func.max(table.c.sun_hours))
    else:
        return query.query(func.max(table.c.sun_hours))


def getMinSunHours(table, se, query=None):
    r"""
    Returns table Data with the least sun hours

    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        return se.query(table, func.min(table.c.sun_hours))
    else:
        return query.query(func.min(table.c.sun_hours))


def getSunHours(sunHours, table, se, query=None):
    r"""
       Method gets all entries with matching amount of sun
       :param sunHours: desired sunhours
       :param table: which weather table is going to be used
       :param se: Session Object containing connection information
       :param query: Query Object which contains SQL query, if empty one will be created
       :returns: Query Object, can be reused for other queries
       """
    if (query == None):
        query = se.query(table).filter(table.c.sun_hours == sunHours)
        return query
    else:
        query = query.filter(table.c.sun_hours == sunHours)
        return query


def getTempAvg(avgTemp, table, se, query=None):
    r"""
       Method gets all entries with desired average temperature
       :param avgTemp: desired average temperature
       :param table: which weather table is going to be used
       :param se: Session Object containing connection information
       :param query: Query Object which contains SQL query, if empty one will be created
       :returns: Query Object, can be reused for other queries
       """

    if(str(table) == "dwd"):
        if (query == None):
            query = se.query(table).filter(table.c.average_temp == avgTemp)
            return query
        else:
            query = query.filter(table.c.average_temp == avgTemp)
            return query
    else:
        if (query == None):
            query = se.query(table).filter(table.c.temp == avgTemp)
            return query
        else:
            query = query.filter(table.c.temp == avgTemp)
            return query


def getMaxPrec(table, se, query=None):
    r"""
    Returns table Data with the most precipitation amount

    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        return se.query(table, func.max(table.c.precipitation_amount))
    else:
        return query.query(func.max(table.c.precipitation_amount))


def getMinPrec(table, se, query=None):
    r"""
    Returns table Data with the least precipitation amount

    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        return se.query(table, func.min(table.c.precipitation_amount))
    else:
        return query.query(func.min(table.c.precipitation_amount))


def getPrecAvg(avgPrec, table, se, query=None):
    r"""
       Method gets all entries with average precipitation higher than/equal to input average precipitation
       :param avgPrec: desired average temperature
       :param table: which weather table is going to be used
       :param se: Session Object containing connection information
       :param query: Query Object which contains SQL query, if empty one will be created
       :returns: Query Object, can be reused for other queries
       """
    if (query == None):
        query = se.query(table).filter(table.c.precipitation_amount >= avgPrec)
        return query
    else:
        query = query.filter(table.c.precipitation_amount >= avgPrec)
        return query


def getMaxWindUp(maxWind, table, se, query=None):
    r"""
    get all data from table with max_wind_speed >= given value
    :param maxWind: max_wind_speed value
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if(str(table) == "dwd"):
        if query is None:
            query = se.query(table).filter(table.c.max_wind_speed >= maxWind)
            return query
        else:
            query = query.filter(table.c.max_wind_speed >= maxWind)
            return query
    else:
        print("No maxwind available!")
        return query


def getPrecType(precipitationType, table, se, query=None):
    r"""
       Method gets all entries with given pecipitation type
       :param precipitationType: desired pecipitation type
       :param table: which weather table is going to be used
       :param se: Session Object containing connection information
       :param query: Query Object which contains SQL query, if empty one will be created
       :returns: Query Object, can be reused for other queries
       """
    if (query == None):
        query = se.query(table).filter(table.c.precipitation_type == precipitationType)
        return query
    else:
        query = query.filter(table.c.precipitation_type == precipitationType)
        return query


def getDate(date, table, se, query=None):
    r"""
       Method gets all entries with given date
       :param postcode: date YYYYMMDD (e.g. 20170512)c
       :param table: which weather table is going to be used
       :param se: Session Object containing connection information
       :param query: Query Object which contains SQL query, if empty one will be created
       :returns: Query Object, can be reused for other queries
       """
    if query is None:
        query = se.query(table).filter(table.c.measure_date == date)
        return query
    else:
        query = query.filter(table.c.measure_date == date)
        return query


def getAvgWindSpeedUp(avgWind, table, se, query=None):
    r"""
    get all data from table with average_wind_speed >= given value
    :param maxWind: max_wind_speed value
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if(str(table) == "dwd"):
        if query is None:
            query = se.query(table).filter(table.c.average_wind_speed >= avgWind)
            return query
        else:
            query = query.filter(table.c.average_wind_speed >= avgWind)
            return query
    else:
        if query is None:
            query = se.query(table).filter(table.c.wind_speed >= avgWind)
            return query
        else:
            query = query.filter(table.c.max_wind_speed >= avgWind)
            return query


def getAvgWindSpeedDown(avgWindSpeed, table, se, query):
    r"""
    get all data from table with average_wind_speed <= given value
    :param avgWindSpeed: average_wind_speed value
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if(str(table) == "dwd"):
        if query is None:
            query = se.query(table).filter(table.c.average_wind_speed <= avgWindSpeed)
            return query
        else:
            query = query.filter(table.c.average_wind_speed <= avgWindSpeed)
            return query
    else:
        if query is None:
            query = se.query(table).filter(table.c.wind_speed <= avgWindSpeed)
            return query
        else:
            query = query.filter(table.c.max_wind_speed <= avgWindSpeed)
            return query


def getSnowHeightUp(snow, table, se, query=None):
    r"""
    get all data from table with snow_height >= given value
    :param snow: snow_height value
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if(str(table) == "dwd"):
        if query is None:
            query = se.query(table).filter(table.c.snow_height >= snow)
            return query
        else:
            query = query.filter(table.c.snow_height >= snow)
            return query
    else:
        print("No snowheigt available!")
        return query


def getMaxTemp(table, se, query=None):
    r"""
    Returns table Data with the highest temperature

    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        return se.query(table, func.max(table.c.max_temp))
    else:
        return query.query(func.max(table.c.max_temp))


def getMaxTempUp(maxTemp, table, se, query=None):
    r"""
    get all data from table with max_temp >= given value
    :param maxTemp: max_temp value
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        query = se.query(table).filter(table.c.max_temp >= maxTemp)
        return query
    else:
        query = query.filter(table.c.max_temp >= maxTemp)
        return query


def getMaxTempDown(maxTemp, table, se, query=None):
    r"""
    get all data from table with max_temp <= given value
    :param maxTemp: max_temp value
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if(query == None):
        query = se.query(table).filter(table.c.max_temp <= maxTemp)
        return query
    else:
        query = query.filter(table.c.max_temp <= maxTemp)
        return query


def getMinTemp(table, se, query=None):
    r"""
    Returns table Data with the lowest temperature

    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        return se.query(table, func.min(table.c.min_temp))
    else:
        return query.query(func.min(table.c.min_temp))


def getMinTempUp(minTemp, table, se, query=None):
    r"""
    Returns table Data for the given Temperature upwards
    :param minTemp: Temperature to sort downwards (>=)
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if(query == None):
        query = se.query(table).filter(table.c.min_temp >= minTemp)
        return query
    else:
        query = query.filter(table.c.min_temp >= minTemp)
        return query


def getMinTempDown(minTemp, table, se, query=None):
    r"""
    Returns table Data for the given Temperature downwards

    :param minTemp: Temperature to sort downwards (<=)
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        query = se.query(table).filter(table.c.min_temp <= minTemp)
        return query
    else:
        query = query.filter(table.c.min_temp <= minTemp)
        return query


def getCoverageUp(coverage, table, se, query=None):
    r"""
    Returns table Data for the given Coverage upwards

    :param coverage: Coverage to sort upwards (>=)
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        query = se.query(table).filter(table.c.coverage_amount >= coverage)
        return query
    else:
        query = query.filter(table.c.coverage_amount >= coverage)
        return query


def getCoverageDown(coverage, table, se, query=None):
    r"""
    Returns table Data for the given Coverage downwards

    :param coverage: Coverage to sort downwards (<=)
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :param query: Query Object which contains SQL query, if empty one will be created
    :returns: Query Object, can be reused for other queries
    """
    if (query == None):
        query = se.query(table).filter(table.c.coverage_amount <= coverage)
        return query
    else:
        query = query.filter(table.c.coverage_amount <= coverage)
        return query

def getColumnList(columnlist, table, se):
    r"""
    if columnlist is not specified: query on the whole table,
    else: only display given columns
    <!> Must be called before calling other queries

    :param columnlist:  list of column names that should be displayed, eg: ["station_id", "station_name", "postcode", "precipitation_amount"]
    :param table: which weather table is going to be used
    :param se: Session Object containing connection information
    :returns: Query Object, can be reused for other queries
    """
    if not columnlist:
        return se.query(table)
    parameters = []
    for col in columnlist:
        parameters.append(getattr(table.c, col))
    return se.query(*parameters)


# Finish query and return
########################
# Session close hier!?###
########################
def getResult(query, se):
    r"""
    :param query: Query Object which contains SQL query
    :param se: Session Object containing the connection information
    :returns: NumpyArray containing all information
    """
    result = []
    for a in query:
        result.append(a)
    if result == []:
        se.close()
        return "Result was empty"
    else:
        se.close()
        return np.vstack(result)


def read_username_password(path_to_login_file):
    file = open(path_to_login_file, 'r')
    username = (file.readline()[10:-1])
    password = file.readline()[10:]
    file.close()
    return username,password


#Setup Data
def getConnectionData(tablename):
    try:
        username, password = read_username_password(path_to_login_file)
        Base = declarative_base()
        engine = create_engine(
            'mysql+pymysql://'+username+':'+password+'@weather.service.tu-berlin.de/dwdtest?use_unicode=1&charset=utf8&ssl_cipher=AES128-SHA')
        # mysql --ssl-cipher=AES128-SHA -u dwdtestuser -p -h weather.service.tu-berlin.de dwdtest
        # mysql --ssl -u dwdtestuser -p -h weather.service.tu-berlin.de dwdtest
        Base.metadata.create_all(engine)
        Session = sqla.orm.sessionmaker()
        Session.configure(bind=engine)
        metadata = MetaData(engine, reflect=True)
        se = Session()
        # connect to database
    except Exception as error:
        if "Engine not found" in str(error):
            print("Engine not found")
        else:
            print(error)
    print("Retrieve table '"+tablename+"' from database:")
    tables = []
    for t in metadata.tables.keys():
        tables.append(t)
    print(tables)
    # Get Table
    table = metadata.tables[tablename]

    return (table, se)

def getConnectionDWD():
    return getConnectionData('dwd')

def getConnectionWetterdienstde():
    return getConnectionData('wetterdienstde')

def getConnectionWetterde():
    return getConnectionData('wetterde')

def getConnectionWettercom():
    return getConnectionData('wettercom')

def getConnectionOpenWeatherMaporg():
    return getConnectionData('openweathermaporg')

def getConnectionAccuweathercom():
    return getConnectionData('accuweathercom')

def getConnectionTest():
    return getConnectionData('testwebsite')

####################################################

def main():
    r"""
    TESTFUNCTION DO NOT USE
    """
    Dwd, se = getConnectionDWD()

    # # Test new functions
    print()
    print("Compare average temperature")
    print(compare_average_temp(Dwd, getConnectionWettercom()[0], se))

    print()
    print("Compare sun_hours")
    print(compare_sun_hours(Dwd, getConnectionWettercom()[0], se))

    print()
    print("Compare wind_speed")
    print(compare_wind_speed(Dwd, getConnectionWettercom()[0], se))

    print()
    print("Compare air_pressure")
    print(compare_air_pressure(Dwd, getConnectionWettercom()[0], se))

    print()
    print("Compare clouds")
    print(compare_clouds(Dwd, getConnectionOpenWeatherMaporg()[0], se))

    print()
    print("Compare humidity")
    print(compare_humidity(Dwd, getConnectionOpenWeatherMaporg()[0], se))

    print()
    print("Filter auf Postleitzahl danach auf avg_temp")
    print(getResult(getTempAvg(0, Dwd, se, getPostcode(26197, Dwd, se)), se))

    print()
    print("Postcode")
    print(getResult(getPostcode(26197, Dwd, se), se))

    print()
    print("Postcode with fewer columns")
    print(getResult(getPostcode(26197,Dwd,se, getColumnList(["station_id", "station_name", "postcode", "precipitation_amount"], Dwd, se)),se))

    print()
    print("Rain")
    print(getResult(getRain(Dwd, se), se))

    print()
    print("SunHours")
    print(getResult(getSunHours(12, Dwd, se), se))

    print()
    print("TempAvg")
    print(getResult(getTempAvg(22, Dwd, se), se))

    print()
    print("Postcode on Rain")
    print(getResult(getRain(Dwd, se, getPostcode(26197, Dwd, se)), se))

    print()
    print("Postcode on SunHours on tempAvg")
    print(getResult(getTempAvg(22, Dwd, se, getSunHours(12, Dwd, se, getPostcode(26197, Dwd, se))), se))

if __name__ == '__main__':
    main()