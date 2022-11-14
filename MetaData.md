# Data Description


**FLIGHTS**
**Note:** All periods of time are expressed in minutes

* YEAR: Year of the Flight Trip
* MONTH: Month of the Flight Trip
* DAY: Day of the Flight Trip
* DAY_OF_WEEK: Day of week of the Flight Trip
* AIRLINE: AIrline Identifier
* FLIGHT_NUMBER: Flight Identifier
* TAIL_NUMBER: Aircraft Identifier
* ORIGIN_AIRPORT: Starting Airport
* DESTINATION_AIRPORT: Destination Airport
* SCHEDULED_DEPARTURE: Planned Departure Time
* DEPARTURE_TIME: Actual Departure Time
* DEPARTURE_DELAY: Total Delay on Departure (DEPARTURE_TIME - SCHEDULED_DEPARTURE)
* TAXI_OUT: The time duration elapsed between departure from the origin airport gate and wheels off
* WHEELS_OFF: time that an aircraft lifts off from the origin airport
* SCHEDULED_TIME: Planned time amount needed for the flight trip 
* ELAPSED_TIME: Actual flight trip duration (AirTime + TaxiIn + TaxiOut)
* AIR_TIME: The time duration between wheels_off and wheels_on time
* DISTANCE: Distance between two airports
* WHEELS_ON: the actual time of departure where operations is concerned.
* TAXI_IN: The time duration elapsed between wheels-on and gate arrival at the destination airport
* SCHEDULED_ARRIVAL: Planned Arrival Time 
* ARRIVAL_TIME: Actual Arrival Time
* ARRIVAL_DELAY: Total Delay on Arrival (ARRIVAL_TIME - SCHEDULED_ARRIVAL)
* DIVERTED: Aircraft landed on airport that out of schedule (1 = diverted) 
* CANCELLED: Flight Cancelled (1 = cancelled)
* CANCELLATION_REASON: N=None and A,B,C are the different codes
* AIR_SYSTEM_DELAY: Delay caused by air system
* SECURITY_DELAY: Delay caused by security
* AIRLINE_DELAY: Delay caused by the airline
* LATE_AIRCRAFT_DELAY: Delay caused by aircraft
* WEATHER_DELAY: Delay caused by weather



**AIRLINES**
* IATA_CODE: código que identifica a una aerolínea 
* AIRLINE: nombre de la aerolínes



**AIRPORTS**
IATA_CODE: código del aeropuerto
AIRPORT: nombre del aeropuerto
CITY: ciudad en la que se encuentra el aeropuerto
STATE: estado en el que se encuentra el aeropuerto
COUNTRY: USA
LATITUDE: latitud del aeropuerto
LONGITUDE: altitud del aeropuerto