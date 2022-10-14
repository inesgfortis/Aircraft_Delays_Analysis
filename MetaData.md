# Data Description

**Note: All periods of time are in minutes**

* Index: Unique Key
* Year: Year of the Flight Trip
* Month: Month of the Flight Trip
* DayofMonth: Day of the Flight Trip
* DayOfWeek: Day of week of the Flight Trip
* DepTime: Actual Departure Time
* CRSDepTime: Planned Departure Time
* ArrTime: Actual Arrival Time
* CRSArrTime: Planned Arrival Time
* UniqueCarrier: Airline Identifier
* FlightNum: Flight Identifier
* TailNum: Aircraft Identifier
* ActualElapsedTime: Actual flight trip duration (AirTime + TaxiIn + TaxiOut)
* CRSElapsedTime: Planned time amount needed for the flight trip
* AirTime: The time duration between wheels_off and wheels_on time
* ArrDelay: Total Delay on Departure (DepTime - CRSDepTime)
* DepDelay: Total Delay on Arrival (ArrTime - CRSArrTime)
* Origin: Starting Airport
* Dest: Destination Airport
* Distance: Distance between two airports
* TaxiIn: The time duration elapsed between wheels-on and gate arrival at the destination airport
* TaxiOut: The time duration elapsed between departure from the origin airport gate and wheels off
* Cancelled: Flight Cancelled (1 = cancelled)
* CancellationCode: N=None and A,B,C are the different codes
* Diverted: Aircraft landed on airport that out of schedule (1 = diverted)
* CarrierDelay: Delay caused by the airline
* WeatherDelay: Delay caused by weather
* NASDelay: Delay caused by air system
* SecurityDelay: Delay caused by security
* LateAircraftDelay: Delay caused by aircraft
