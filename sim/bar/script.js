

$('#submit').on('click', (e) => {
  e.preventDefault();
  let input=$('textarea').val();
  input=input.split('\n');
  console.log(input);
  dataArray=input.map(item=>{
    let itemParts=item.split(',');
    return (parseFloat(itemParts[1].replace(/\s/g, '')));
  })
  labelArray=input.map(item=>{
    let itemParts=item.split(',');
    return (itemParts[0]);
  })
  console.log(dataArray, labelArray);
  makeBarChart(dataArray, labelArray);
})


$(document).ready(function () {
  let dataset = [8, 10, 5, 12, 18, 3, 4, 12, 16];
  let labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"];
  makeBarChart(dataset, labels);
});


function makeBarChart(data, label) {

  $('svg').empty();
 
  let dataset=data;
  label.unshift("");
  let labels=label;
  console.log(label);

  let svgWidth = 500,
    svgHeight = 300,
    barPadding = 5;
  let barWidth = (svgWidth / dataset.length);
  let marg = 50;


  let svg = d3.select('svg')
    .attr("width", svgWidth + marg * 2)
    .attr("height", svgHeight + marg * 2);


  let xScale = d3.scaleLinear()
    .domain([0, (dataset.length) + 1])
    .range([0, svgWidth + barWidth]);

  let yScale = d3.scaleLinear()
    .domain([0, d3.max(dataset)])
    .range([svgHeight, 0]);

  let x_axis = d3.axisBottom()
    .tickFormat(d => {
      return labels[d];
    })
    .scale(xScale);

  let y_axis = d3.axisLeft()
    .scale(yScale);

  let barChart = svg.selectAll("rect")
    .data(dataset)
    .enter()
    .append("rect")
    .attr('data-num', (d, i) => i)
    .attr('fill', '#9bbcf2')
    .attr("y", (d) => yScale(d) + marg)
    .attr("height", (d) => yScale(0) - yScale(d))
    .attr("width", barWidth - barPadding)
    .attr("transform", (d, i) => {
      let translate = [marg + barWidth * i, 0];
      return "translate(" + translate + ")";
    });

  let text = svg.selectAll("text")
    .data(dataset)
    .enter()
    .append("text")
    .attr('fill', '#636363')
    .attr('font-size','10px')
    .attr('data-num2', (d, i) => i)
    .attr('visibility', 'hidden')
    .text((d,i) => {return labels[i+1]+" "+d})
    .attr("y", (d, i) => marg + yScale(d) - 10)
    .attr("x", (d, i) => (marg + barWidth * i ));


  d3.selectAll("rect").on("mouseover", mouseover).on("mouseout", mouseout);


  let htrans = 30 - barPadding / 2;
  svg.append("g")
    .attr("transform", `translate(${htrans}, ${marg})`)
    .call(y_axis);

  let vtrans2 = svgHeight + marg;
  let htrans2 = -barWidth / 2 - barPadding / 2 + marg;
  svg.append("g")
    .attr("transform", `translate(${htrans2},${vtrans2})`)
    .call(x_axis);


}

function mouseover() {
  d3.select(this).attr("opacity", .5);
  dat = d3.select(this).attr('data-num');
  t = d3.selectAll("[data-num2='" + dat + "']");
  t.attr('visibility', 'visible');

}

function mouseout() {
  d3.select(this).attr("opacity", 1);
  dat = d3.select(this).attr('data-num');
  t = d3.selectAll("[data-num2='" + dat + "']");
  t.attr('visibility', 'hidden');
}