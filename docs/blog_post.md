# The Boundary Detection Project - Part #1 - A Revisit to My Digital Image Processing Final Project Assignment 

Internet of Things (IoT) and Computer Vision (CV) can become a dangerous tool together. If you don't believe me, go read 
1984 from George Orwell and meet the infamous Big Brother. There is always the bright side of the moon though. This post
will be about a revisit to one of the projects I did during my Master's in Bogazici University. In one of the most
entertaining courses I have taken -Digital Image Processing course from the Electrical \& Electronics Engineering Department,
I took on the project to detect boundaries between products in a supermarket shelf image. So a natural question to ask is,
what does that have to do with IoT and CV?

IoT and CV together can make a great impact on our everyday business making. Checking out examples is the easiest to grasp 
the extend of what can be done with them. I recommend you to take a look at the **Interesting Use Cases** section of this 
[blog post](https://www.iotforall.com/computer-vision-iot/) by Frank Lee. For now, I want you to focus on another 
use case, namely *smart shelves* (or *intelligent shelves*) in a retail setting. There are many ways to make a shelf
*smart* but I would like to focus on CV related solutions. Take a look at [Trax's Shelf Pulse product](https://traxretail.com/products/shelf-pulse/).
It's an elegant example of a pipeline all the way from data at nodes to dashboards and insights at the platform.  

## What's the technology?

The basic idea is to automate the process of reporting which item is placed on which part of a specific shelf in a
specific store. Traditionally, this would take quite a long time for a worker in the store to do. The worker has to
go through each shelf, manually observe and count all the products and identify missing/misplaced items by cross-checking 
[plangorams](https://www.thebalancesmb.com/retail-planograms-2890336) to eventually report everything down. With the 
recent advancements in CV algorithms and careful engineering, this manual step can be bypassed. Workers can use mobile cameras
to take pictures of the shelves and/or fixed cameras can constantly monitor certain parts of the shelves to collect images
to be processed by at the cloud platform. The same platform can immediately generate reports or other forms of useful 
information to help business achieve its goals. If you check the [Products section of Trax's website](https://traxretail.com/products/), you can see
many different tools that enhance various business processes for both retail stores and CPG companies. 
In order to get an idea about how much value is perceived to be in their products, check out their [funding history](https://www.crunchbase.com/organization/traxretail#section-funding-rounds). 
We're talking about hundreds of millions here.

[//]: # (TODO: put a photo from https://arxiv.org/pdf/1707.08378.pdf here, explain it above in the last sentence)

## The Project

In this post, I will explain a very basic sub-problem of the whole pipeline: detecting split lines between products on
the shelves. Why would you care? Remember our end goal was to identify product instances in each shelf. The idea is that
if we recover the locations of the split lines and assume that there is at most one product between those lines, we will
greatly reduce the search space of our object recognition algorithm. In other words, we will only have to look for one
and only one object between each pair of consecutive split line (assuming the absence of a product is assumed to be another class).
This provides a very strong prior on our object recognition algorithm, possibly making it faster and much more accurate.
If you think about it, this step is analogous to the Region Proposal Network(RPN) stage of [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) 
or [Mask R-CNN](https://arxiv.org/pdf/1703.06870.pdf). We just simply exploit the inherent geometrical structure of the
problem to create a marginal advantage over more generalized algorithms by making our own customized version.

[//]: # (TODO: whole picture to shelves, shelves to split lines, split lines to )

### What kind of a data we have?

  