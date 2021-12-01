d3.selectAll("button").on("click", function () {
    // error handling

    var Suburb = d3.select("#Suburb").property("value");
    var Address = d3.select("#Address").property("value");
    var Rooms = d3.select("#Rooms").property("value");
    var Type = d3.select("#Type").property("value");
    var Price = d3.select("#Price").property("value");
    var Distance = d3.select("#Distance").property("value");
    var Bedroom2 = d3.select("#Bedroom2").property("value");
    var Bathroom = d3.select("#Bathroom").property("value");
    var Car = d3.select("#Car").property("value");
    var Landsize = d3.select("#Landsize").property("value");
    var CouncilArea = d3.select("#CouncilArea").property("value");
    var Regionname = d3.select('input[name="Regionname"]:checked').node().value

    
    console.log(Suburb, Address, Rooms, Type, Price, Distance, Postcode, Bedroom2, Bathroom, Car, Landsize, CouncilArea, Regionname);
    
    var data = {
        Suburb : Suburb,
        Address : Address,
        Rooms : Type, 
        Price : Price, 
        Distance : Distance, 
        Postcode : Postcode, 
        Bedroom2 : Bedroom2, 
        Bathroom, 
        Car : Car, 
        Landsize : Landsize, 
        CouncilArea : CouncilArea, 
        Regionname : Regionname
        
    }
    console.log(data)

   

});