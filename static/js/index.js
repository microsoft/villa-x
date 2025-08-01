window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
  // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    // ✅ Options for 3-video carousels
    const options3 = {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: true,
      autoplay: false,
      autoplaySpeed: 3000,
    };

    // ✅ Options for 1-video carousels
    const options1 = {
      slidesToScroll: 1,
      slidesToShow: 1,
      loop: true,
      infinite: true,
      autoplay: false,
      autoplaySpeed: 3000,
    };

    // ✅ Attach each class with different config
    const carousels3 = bulmaCarousel.attach('.carousel-threevideo', options3);
    const carousels1 = bulmaCarousel.attach('.carousel-onevideo', options1);

    [...carousels3, ...carousels1].forEach(carousel => {
      carousel.on('before:show', state => {
        console.log(state);
      });
    });

    bulmaSlider.attach();
})