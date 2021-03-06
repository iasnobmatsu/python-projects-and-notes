$('#submit').on('click', (e) => {
  e.preventDefault();
  let n = parseInt($('#n').val());
  let p = parseFloat($('#p').val());
  makeBinomial(n, p);
});

$('#calc').on('click', (e) => {
  e.preventDefault();
  let x = parseInt($('#x-val').val());
  let n = parseInt($('#n2').val());
  let p = parseFloat($('#p2').val());
  let result = (factorial(n) / (factorial(x) * factorial(n - x))) * Math.pow(p, x) * Math.pow((1 - p), (n - x));
  $('#bin-result').html(`P(${x})=${result}`);
})


$(document).ready(function () {
  makeBinomial(4, 0.5);
});

function factorial(num) {
  if (num == 0) {
    return 1;
  } else {
    return (num * factorial(num - 1));
  }
}

function combination(n,k) {//n!/(k!*(n-k)!)
  if (n==k || k==0) {
    return 1;
  } else {
    k=Math.max(k,n-k);
    let num=k+1;
    for (let i=num;i<=n;i++){
      num=num*i;
    }
    let den=1;
    for (let i=1; i<=(n-k);i++){
      den=den*i;
    }
    return num/den;
  }
}


function makeBinomial(n, p) {

  $('svg').empty();
  let labels = [];
  for (let i = 0; i <= n; i++) {
    labels.push(i);
  }
  let dataset = [];
  for (let i = 0; i <= n; i++) {
    dataset.push((factorial(n) / (factorial(i) * factorial(n - i))) * Math.pow(p, i) * Math.pow((1 - p), (n - i)));
  }

  labels.unshift("");
  let svgWidth = 500,
    svgHeight = 300,
    barPadding = 1;
  let barWidth = (svgWidth / dataset.length);
  if (barWidth <= barPadding) {
    barPadding = 0;
  }
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
    .attr("height", (d) =>  yScale(0) - yScale(d))
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
    .attr('font-size', '10px')
    .attr('data-num2', (d, i) => i)
    .attr('visibility', 'hidden')
    .text((d, i) => {
      return labels[i + 1] + " " + d
    })
    .attr("y", (d, i) => marg + yScale(d) - 10)
    .attr("x", (d, i) => (marg + barWidth * i));


  d3.selectAll("rect").on("mouseover", mouseover).on("mouseout", mouseout);


  let htrans = 40 - barPadding / 2;
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